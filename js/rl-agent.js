/**
 * rl-agent.js — агент с Deep Q-Learning.
 *
 * Реализует:
 *  - Double DQN (van Hasselt et al., 2015)
 *  - Prioritized Experience Replay (Schaul et al., 2015)
 *  - Adam optimizer через AdvancedNeuralNetwork
 *  - Dueling DQN (опционально, Wang et al., 2016)
 */
import { Agent, INPUT_SIZE, OUTPUT_SIZE } from './agent.js';
import { AdvancedNeuralNetwork } from './neural-network-advanced.js';
import { DuelingNetwork } from './dueling-network.js';

/** Максимальный размер буфера памяти */
const MEMORY_SIZE = 5000;
/** Коэффициент дисконтирования наград */
const GAMMA = 0.95;
/** Начальное значение epsilon */
const EPSILON_START = 1.0;
/** Минимальное значение epsilon */
const EPSILON_MIN = 0.05;
/** Скорость затухания epsilon */
const EPSILON_DECAY = 0.995;
/** Шаг обновления целевой сети */
const TARGET_UPDATE_FREQ = 100;

// ── Константы умных детекторов завершения эпизода ────────────────────────────

/** Размер окна последних позиций для обнаружения зацикливания */
const RECENT_POSITIONS_WINDOW = 20;
/** Максимальное число уникальных клеток в окне, считающееся зациклом */
const LOOP_UNIQUE_CELLS_MAX = 3;
/** Доля «челночных» движений (A→B→A), считающаяся зациклом */
const SHUTTLE_RATE_THRESHOLD = 0.7;
/** Размер окна истории расстояний до цели */
const DISTANCE_HISTORY_WINDOW = 20;
/** Интервал в шагах между записями расстояния до цели */
const DISTANCE_SAMPLE_INTERVAL = 10;
/** Минимальный прогресс к цели (снижение расстояния) за окно истории */
const PROGRESS_THRESHOLD = 0.95;
/** Минимальное число шагов перед проверкой прогресса */
const MIN_STEPS_FOR_PROGRESS = 200;
/** Порог доли повторных посещений, считающийся избыточным */
const REPEAT_RATE_THRESHOLD = 0.70;
/** Минимальное число шагов перед проверкой повторов */
const MIN_STEPS_FOR_REPEAT = 100;
/** Аварийный лимит шагов (страховка от зависания) */
const EMERGENCY_STEP_LIMIT = 10000;

export class RLAgent extends Agent {
    /**
     * @param {number|number[]} hiddenLayers - нейронов в скрытом слое или массив [64,32]
     * @param {object} [options]
     * @param {boolean} [options.doubleDQN=true]           - Double DQN
     * @param {boolean} [options.prioritizedReplay=true]   - приоритизированный replay
     * @param {boolean} [options.dueling=false]            - Dueling DQN
     * @param {number}  [options.learningRate=0.001]
     * @param {boolean} [options.useBatchNorm=false]
     * @param {number}  [options.dropoutRate=0.0]
     */
    constructor(hiddenLayers = 16, options = {}) {
        // Поддержка передачи одного числа (legacy) или массива
        const layers = Array.isArray(hiddenLayers) ? hiddenLayers : [hiddenLayers];
        super(null, layers[0]);

        const {
            doubleDQN         = true,
            prioritizedReplay = true,
            dueling           = false,
            learningRate      = 0.001,
            useBatchNorm      = false,
            dropoutRate       = 0.0,
        } = options;

        this.doubleDQN         = doubleDQN;
        this.prioritizedReplay = prioritizedReplay;

        const netOptions = { learningRate, useBatchNorm, dropoutRate };
        const NetClass   = dueling ? DuelingNetwork : AdvancedNeuralNetwork;

        /** Основная Q-сеть */
        this.brain = new NetClass(INPUT_SIZE, layers, OUTPUT_SIZE, netOptions);

        /** Целевая Q-сеть (обновляется реже) */
        this.targetBrain = this.brain.copy();

        /** Буфер опыта */
        this.memory = [];

        /** TD-ошибки для приоритизированного replay */
        this.priorities = [];

        /** Текущая сумма приоритетов (для O(1) выборки) */
        this._totalPriority = 0;

        /** Степень приоритизации (α) */
        this.alpha = 0.6;

        /** Компенсация смещения importance sampling (β, растёт до 1.0) */
        this.beta = 0.4;
        this.betaIncrement = 0.001;

        /** Счётчик шагов (для обновления целевой сети) */
        this.totalSteps = 0;

        /** Текущий epsilon */
        this.epsilon = EPSILON_START;

        /** Суммарные награды за эпизод */
        this.episodeReward = 0;

        /** История наград по эпизодам */
        this.rewardHistory = [];

        /** Глобальная карта посещений (не сбрасывается между эпизодами) */
        this.globalVisited = new Set();
        this.globalVisitCount = new Map();

        /** Счётчик эпизодов для статистики */
        this.episodeCount = 0;

        /** История эффективности исследования (% новых клеток за эпизод) */
        this.explorationEfficiency = [];

        // ── Поля умных детекторов завершения эпизода ─────────────────────────

        /** Последние N позиций для обнаружения зацикливания */
        this.recentPositions = [];
        /** Размер окна для recentPositions */
        this.maxRecentPositions = RECENT_POSITIONS_WINDOW;

        /** История расстояний до цели (записывается каждые 10 шагов) */
        this.distanceHistory = [];
        /** Размер окна для distanceHistory */
        this.maxDistanceHistory = DISTANCE_HISTORY_WINDOW;

        /** Предыдущая позиция агента */
        this.prevPos = null;

        /** Статистика причин завершения эпизодов */
        this.terminationReasons = {
            success:    0,
            stuck:      0,
            noProgress: 0,
            repetitive: 0,
            emergency:  0,
        };

        /** Общее количество эпизодов */
        this.totalEpisodes = 0;
        /** Количество успешных эпизодов */
        this.successfulEpisodes = 0;
    }

    // ── Умные детекторы завершения эпизода ───────────────────────────────────

    /**
     * Обнаружить зацикливание агента.
     * Проверяет три паттерна:
     * 1. Малое число уникальных позиций в окне
     * 2. Повторяющаяся последовательность позиций
     * 3. «Челночные» движения A→B→A→B
     * @returns {boolean}
     */
    isStuckInLoop() {
        if (this.recentPositions.length < this.maxRecentPositions) {
            return false;
        }

        // Метод 1: мало уникальных позиций в окне
        const uniqueRecent = new Set(this.recentPositions);
        if (uniqueRecent.size <= LOOP_UNIQUE_CELLS_MAX) {
            return true;
        }

        // Метод 2: паттерн из первых 5 позиций повторяется сразу после себя
        const patternLength = 5;
        const pattern = this.recentPositions.slice(0, patternLength);
        const next     = this.recentPositions.slice(patternLength, patternLength * 2);
        if (next.length === patternLength && pattern.join('|') === next.join('|')) {
            return true;
        }

        // Метод 3: более 70% шагов — «челночные» (pos[i] === pos[i-2])
        let shuttleCount = 0;
        for (let i = 2; i < this.recentPositions.length; i++) {
            if (this.recentPositions[i] === this.recentPositions[i - 2]) {
                shuttleCount++;
            }
        }
        if (shuttleCount / (this.recentPositions.length - 2) > SHUTTLE_RATE_THRESHOLD) {
            return true;
        }

        return false;
    }

    /**
     * Обнаружить отсутствие прогресса к цели.
     * Сравнивает среднее расстояние в первой и второй половине истории.
     * @returns {boolean}
     */
    isNotProgressing() {
        if (this.steps < MIN_STEPS_FOR_PROGRESS ||
            this.distanceHistory.length < this.maxDistanceHistory) {
            return false;
        }

        const halfLength = Math.floor(this.distanceHistory.length / 2);
        const firstHalf  = this.distanceHistory.slice(0, halfLength);
        const secondHalf = this.distanceHistory.slice(-halfLength);

        const avgFirst  = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
        const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

        // Нет прогресса: расстояние не уменьшилось хотя бы на 5%
        return avgSecond >= avgFirst * PROGRESS_THRESHOLD;
    }

    /**
     * Обнаружить избыточные повторные посещения клеток.
     * Если более 70% шагов — повторные визиты, агент неэффективен.
     * @returns {boolean}
     */
    isTooRepetitive() {
        if (this.steps < MIN_STEPS_FOR_REPEAT) {
            return false;
        }

        const uniqueCells    = this.visited.size;
        const repeatedVisits = this.steps - uniqueCells;
        const repeatRate     = repeatedVisits / this.steps;

        return repeatRate > REPEAT_RATE_THRESHOLD;
    }

    /**
     * Проверить, завершён ли эпизод.
     * Использует набор умных детекторов вместо жёсткого maxSteps.
     * @returns {boolean}
     */
    isDone() {
        // 1. Успешное достижение цели
        if (this.reached) {
            this.terminationReasons.success++;
            console.log(`✅ Эпизод ${this.episodeCount}: Цель достигнута за ${this.steps} шагов!`);
            return true;
        }

        // 2. Застрял в цикле
        if (this.isStuckInLoop()) {
            this.terminationReasons.stuck++;
            console.warn(`🔄 Эпизод ${this.episodeCount}: Зацикливание обнаружено (${this.steps} шагов, ${this.visited.size} уникальных клеток)`);
            return true;
        }

        // 3. Нет прогресса к цели
        if (this.isNotProgressing()) {
            this.terminationReasons.noProgress++;
            console.warn(`📉 Эпизод ${this.episodeCount}: Нет прогресса к цели (${this.steps} шагов, расстояние: ${this._distToGoal().toFixed(1)})`);
            return true;
        }

        // 4. Слишком много повторных посещений
        if (this.isTooRepetitive()) {
            this.terminationReasons.repetitive++;
            const repeatRate = ((this.steps - this.visited.size) / this.steps * 100).toFixed(0);
            console.warn(`🔁 Эпизод ${this.episodeCount}: Избыточные повторы (${this.steps} шагов, ${repeatRate}% повторов)`);
            return true;
        }

        // 5. Аварийный лимит (страховка от зависания)
        if (this.steps >= EMERGENCY_STEP_LIMIT) {
            this.terminationReasons.emergency++;
            console.error(`⚠️ Эпизод ${this.episodeCount}: АВАРИЙНОЕ ЗАВЕРШЕНИЕ (${EMERGENCY_STEP_LIMIT} шагов)!`);
            return true;
        }

        return false;
    }

    // ── Глобальная память ─────────────────────────────────────────────────────

    /**
     * Сбросить состояние эпизода, НО сохранить глобальную память и статистику.
     * @param {{x:number,y:number}} startPos
     */
    reset(startPos) {
        // 1. Сохраняем данные, которые не должны сбрасываться
        const globalVisited       = this.globalVisited;
        const globalVisitCount    = this.globalVisitCount;
        const episodeCount        = this.episodeCount;
        const terminationReasons  = this.terminationReasons;
        const totalEpisodes       = this.totalEpisodes;
        const successfulEpisodes  = this.successfulEpisodes;
        const wasReached          = this.reached;

        // 2. Вызываем родительский reset (стирает локальную память)
        super.reset(startPos);

        // 3. Восстанавливаем глобальные данные
        this.globalVisited      = globalVisited;
        this.globalVisitCount   = globalVisitCount;
        this.terminationReasons = terminationReasons;
        this.episodeCount       = episodeCount + 1;
        this.totalEpisodes      = totalEpisodes + 1;
        this.successfulEpisodes = wasReached ? successfulEpisodes + 1 : successfulEpisodes;

        // 4. Затухание памяти каждые 50 эпизодов
        if (this.episodeCount % 50 === 0) {
            this.decayGlobalMemory(0.7);
        }

        // 5. Сбрасываем детекторы для нового эпизода
        this.recentPositions = [];
        this.distanceHistory = [];
        this.prevPos = { ...startPos };

        // 6. Обнулить награду эпизода
        this.episodeReward = 0;

        // 7. Логируем статистику каждые 10 эпизодов
        if (this.episodeCount % 10 === 0) {
            this.logTerminationStats();
        }
    }

    /**
     * Получить статистику причин завершения эпизодов.
     * @returns {Object}
     */
    getTerminationStats() {
        const total = this.totalEpisodes;
        const stats = {};

        for (const [reason, count] of Object.entries(this.terminationReasons)) {
            stats[reason] = {
                count,
                percent: total > 0 ? ((count / total) * 100).toFixed(1) : '0.0',
            };
        }

        stats.total       = total;
        stats.successRate = total > 0
            ? ((this.successfulEpisodes / total) * 100).toFixed(1)
            : '0.0';

        return stats;
    }

    /**
     * Логировать статистику причин завершения в консоль.
     */
    logTerminationStats() {
        const stats = this.getTerminationStats();
        console.log(
            `\n📊 Статистика после ${stats.total} эпизодов:\n` +
            `   ✅ Успех: ${stats.success.count} (${stats.success.percent}%)\n` +
            `   🔄 Циклы: ${stats.stuck.count} (${stats.stuck.percent}%)\n` +
            `   📉 Нет прогресса: ${stats.noProgress.count} (${stats.noProgress.percent}%)\n` +
            `   🔁 Повторы: ${stats.repetitive.count} (${stats.repetitive.percent}%)\n` +
            `   ⚠️ Аварийные: ${stats.emergency.count} (${stats.emergency.percent}%)\n` +
            `\n   🎯 Success rate: ${stats.successRate}%`,
        );
    }

    /**
     * Обновить глобальную память при посещении клетки.
     * @param {number} x
     * @param {number} y
     */
    updateGlobalMemory(x, y) {
        const key = `${x},${y}`;
        this.globalVisited.add(key);
        this.globalVisitCount.set(key, (this.globalVisitCount.get(key) ?? 0) + 1);
    }

    /**
     * Затухание глобальной памяти (уменьшить счётчики посещений).
     * @param {number} factor - коэффициент затухания (0.7 = уменьшить на 30%)
     */
    decayGlobalMemory(factor = 0.5) {
        for (const [key, count] of this.globalVisitCount.entries()) {
            const newCount = Math.floor(count * factor);
            if (newCount === 0) {
                this.globalVisitCount.delete(key);
                this.globalVisited.delete(key);
            } else {
                this.globalVisitCount.set(key, newCount);
            }
        }
    }

    /**
     * Получить статистику глобальной памяти.
     * @returns {{totalCells: number, avgVisits: number, maxVisits: number}}
     */
    getGlobalMemoryStats() {
        const visits = Array.from(this.globalVisitCount.values());
        return {
            totalCells: this.globalVisited.size,
            avgVisits:  visits.length > 0 ? visits.reduce((a, b) => a + b, 0) / visits.length : 0,
            maxVisits:  visits.length > 0 ? Math.max(...visits) : 0,
        };
    }

    /**
     * Переопределяем move() для трекинга позиций, расстояний и глобальной памяти.
     * @param {number} actionIndex
     */
    move(actionIndex) {
        this.prevPos = { ...this.pos };

        super.move(actionIndex);

        // Обновить глобальную память только если движение было успешным
        if (this.pos.x !== this.prevPos.x || this.pos.y !== this.prevPos.y) {
            this.updateGlobalMemory(this.pos.x, this.pos.y);
        }

        // Записываем позицию для обнаружения зацикливания
        const posKey = `${this.pos.x},${this.pos.y}`;
        this.recentPositions.push(posKey);
        if (this.recentPositions.length > this.maxRecentPositions) {
            this.recentPositions.shift();
        }

        // Записываем расстояние до цели каждые DISTANCE_SAMPLE_INTERVAL шагов
        if (this.steps % DISTANCE_SAMPLE_INTERVAL === 0) {
            const dist = this._distToGoal();
            this.distanceHistory.push(dist);
            if (this.distanceHistory.length > this.maxDistanceHistory) {
                this.distanceHistory.shift();
            }
        }
    }

    // ── Experience Replay ──────────────────────────────────────────────────────

    /**
     * Сохранить опыт в буфер.
     * @param {number[]} state
     * @param {number}   action
     * @param {number}   reward
     * @param {number[]} nextState
     * @param {boolean}  done
     */
    remember(state, action, reward, nextState, done) {
        let priority;
        if (this.prioritizedReplay) {
            // Use max existing priority for new experiences (avoids extra forward passes)
            // New experiences should be sampled at least once before deprioritizing
            priority = this._maxPriority ?? 1;
        } else {
            priority = 1;
        }

        this.memory.push({ state, action, reward, nextState, done });
        this.priorities.push(priority);
        this._totalPriority += priority;

        if (this.memory.length > MEMORY_SIZE) {
            this._totalPriority -= this.priorities[0];
            this.memory.shift();
            this.priorities.shift();
        }

        this.episodeReward += reward;
    }

    /**
     * Выбрать действие по epsilon-greedy стратегии.
     * @param {number[]} state
     * @returns {number}
     */
    act(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * OUTPUT_SIZE);
        }
        const qValues = this.brain.predict(state);
        return qValues.indexOf(Math.max(...qValues));
    }

    /**
     * Приоритизированная выборка мини-батча.
     * @param {number} batchSize
     * @returns {{ batch: object[], indices: number[], weights: number[] }}
     */
    _sampleBatch(batchSize) {
        if (!this.prioritizedReplay) {
            // Равномерная выборка
            const indices = new Set();
            while (indices.size < batchSize) {
                indices.add(Math.floor(Math.random() * this.memory.length));
            }
            const idxArr = [...indices];
            return {
                batch:   idxArr.map(i => this.memory[i]),
                indices: idxArr,
                weights: new Array(batchSize).fill(1),
            };
        }

        const totalPriority = this._totalPriority;
        const probs = this.priorities.map(p => p / totalPriority);

        const batch   = [];
        const indices = [];
        const weights = [];

        for (let i = 0; i < batchSize; i++) {
            const idx = this._sampleIndex(probs);
            indices.push(idx);
            batch.push(this.memory[idx]);
            // Importance sampling weight
            const w = Math.pow(this.memory.length * probs[idx], -this.beta);
            weights.push(w);
        }

        // Нормализация весов
        const maxW = Math.max(...weights);
        return {
            batch,
            indices,
            weights: weights.map(w => w / maxW),
        };
    }

    /**
     * Выбрать индекс по вектору вероятностей.
     * @param {number[]} probs
     * @returns {number}
     */
    _sampleIndex(probs) {
        const r = Math.random();
        let cum = 0;
        for (let i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r <= cum) return i;
        }
        return probs.length - 1;
    }

    /**
     * Обучение на мини-батче из буфера памяти.
     * @param {number} [batchSize=32]
     */
    replay(batchSize = 32) {
        if (this.memory.length < batchSize) return;

        const { batch, indices, weights } = this._sampleBatch(batchSize);

        for (let i = 0; i < batch.length; i++) {
            const { state, action, reward, nextState, done } = batch[i];

            let target;
            if (this.doubleDQN) {
                // Double DQN: основная сеть выбирает действие, целевая оценивает Q
                const nextQ      = this.brain.predict(nextState);
                const bestAction = nextQ.indexOf(Math.max(...nextQ));
                const targetQNext = this.targetBrain.predict(nextState)[bestAction];
                target = done ? reward : reward + GAMMA * targetQNext;
            } else {
                target = done ? reward : reward + GAMMA * Math.max(...this.targetBrain.predict(nextState));
            }

            this.brain.trainOnSample(state, action, target, weights[i]);

            // Обновить приоритет
            if (this.prioritizedReplay) {
                const current    = this.brain.predict(state)[action];
                const tdError    = Math.abs(target - current);
                const newPriority = Math.pow(tdError + 1e-6, this.alpha);
                this._totalPriority += newPriority - this.priorities[indices[i]];
                this.priorities[indices[i]] = newPriority;
                if (newPriority > (this._maxPriority ?? 0)) this._maxPriority = newPriority;
            }
        }

        // Увеличить beta (importance sampling correction)
        this.beta = Math.min(1.0, this.beta + this.betaIncrement);

        // Обновить целевую сеть с заданной периодичностью
        this.totalSteps++;
        if (this.totalSteps % TARGET_UPDATE_FREQ === 0) {
            this.targetBrain = this.brain.copy();
        }
    }

    /**
     * Уменьшить epsilon и записать награду эпизода.
     */
    updateEpsilon() {
        this.rewardHistory.push(this.episodeReward);
        this.episodeReward = 0;

        if (this.epsilon > EPSILON_MIN) {
            this.epsilon *= EPSILON_DECAY;
        }
    }
}
