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
/** Отступ от лимита шагов для определения успешного эпизода в статистике */
const SUCCESS_THRESHOLD_MARGIN = 10;

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

        // ── Адаптивный maxSteps ───────────────────────────────────────────────

        /** Начальный лимит шагов (для исследования) */
        this.baseMaxSteps = 2000;
        /** Минимальный лимит шагов (когда научился) */
        this.minMaxSteps = 300;
        /** Текущий лимит шагов */
        this.currentMaxSteps = this.baseMaxSteps;

        /** Подряд успешных эпизодов */
        this.successStreak = 0;
        /** Подряд неудачных эпизодов */
        this.failureStreak = 0;
        /** Всего успехов */
        this.totalSuccesses = 0;
        /** Всего эпизодов */
        this.totalEpisodes = 0;

        /** Длина последних 20 эпизодов */
        this.episodeStepsHistory = [];
        /** Размер окна истории эпизодов */
        this.maxHistory = 20;
    }

    // ── Адаптивный maxSteps ───────────────────────────────────────────────────

    /**
     * Проверить, завершён ли эпизод.
     * Использует адаптивный maxSteps вместо детекторов.
     * @returns {boolean}
     */
    isDone() {
        // ✅ Успешное завершение
        if (this.reached) {
            return true;
        }

        // ✅ Адаптивный лимит шагов
        if (this.steps >= this.currentMaxSteps) {
            return true;
        }

        return false;
    }

    // ── Глобальная память ─────────────────────────────────────────────────────

    /**
     * Сбросить состояние эпизода, сохранив глобальную память.
     * Адаптировать maxSteps на основе результата эпизода.
     * @param {{x:number,y:number}} startPos
     */
    reset(startPos) {
        // 1. Сохраняем данные, которые НЕ должны сбрасываться
        const globalVisited        = this.globalVisited;
        const globalVisitCount     = this.globalVisitCount;
        const baseMaxSteps         = this.baseMaxSteps;
        const minMaxSteps          = this.minMaxSteps;
        const currentMaxSteps      = this.currentMaxSteps;
        const successStreak        = this.successStreak;
        const failureStreak        = this.failureStreak;
        const totalSuccesses       = this.totalSuccesses;
        const totalEpisodes        = this.totalEpisodes;
        const episodeStepsHistory  = this.episodeStepsHistory;
        const wasReached           = this.reached;
        const episodeCount         = this.episodeCount;

        // Записываем длину эпизода в историю (пропускаем эпизод с 0 шагов)
        if (this.steps > 0) {
            episodeStepsHistory.push(this.steps);
            if (episodeStepsHistory.length > this.maxHistory) {
                episodeStepsHistory.shift();
            }
        }

        // 2. Вызываем родительский reset (сбрасывает локальную память)
        super.reset(startPos);

        // 3. Восстанавливаем глобальные данные
        this.globalVisited        = globalVisited;
        this.globalVisitCount     = globalVisitCount;
        this.baseMaxSteps         = baseMaxSteps;
        this.minMaxSteps          = minMaxSteps;
        this.episodeStepsHistory  = episodeStepsHistory;

        // 4. Затухание памяти каждые 50 эпизодов
        if ((episodeCount + 1) % 50 === 0) {
            this.decayGlobalMemory(0.7);
        }

        // 5. ✅ Адаптируем maxSteps на основе результата
        this.totalEpisodes = totalEpisodes + 1;

        if (wasReached) {
            // ✅ Успех — увеличиваем streak, уменьшаем лимит
            this.totalSuccesses = totalSuccesses + 1;
            this.successStreak  = successStreak + 1;
            this.failureStreak  = 0;

            // Уменьшаем лимит после каждых 3 успехов подряд
            if (this.successStreak % 3 === 0) {
                this.currentMaxSteps = Math.max(
                    minMaxSteps,
                    Math.floor(currentMaxSteps * 0.85),  // -15%
                );
                console.log(`✅ ${this.successStreak} успехов подряд! MaxSteps → ${this.currentMaxSteps}`);
            } else {
                this.currentMaxSteps = currentMaxSteps;
            }
        } else {
            // ❌ Неудача — увеличиваем failureStreak
            this.totalSuccesses = totalSuccesses;
            this.successStreak  = 0;
            this.failureStreak  = failureStreak + 1;

            // Увеличиваем лимит после 10 неудач подряд
            if (this.failureStreak >= 10) {
                this.currentMaxSteps = Math.min(
                    baseMaxSteps,
                    Math.floor(currentMaxSteps * 1.2),  // +20%
                );
                this.failureStreak = 0;
                console.warn(`⚠️ 10 неудач подряд. MaxSteps → ${this.currentMaxSteps}`);
            } else {
                this.currentMaxSteps = currentMaxSteps;
            }
        }

        this.episodeCount  = episodeCount + 1;
        this.episodeReward = 0;

        // 6. Логируем статистику каждые 10 эпизодов
        if (this.episodeCount % 10 === 0) {
            this.logStats();
        }
    }

    /**
     * Логировать статистику обучения.
     */
    logStats() {
        const successRate = this.totalEpisodes > 0
            ? ((this.totalSuccesses / this.totalEpisodes) * 100).toFixed(1)
            : '0.0';

        const avgSteps = this.episodeStepsHistory.length > 0
            ? (this.episodeStepsHistory.reduce((a, b) => a + b, 0) / this.episodeStepsHistory.length).toFixed(0)
            : '0';

        const recentSuccesses = this.episodeStepsHistory.slice(-10).filter(steps =>
            steps < this.currentMaxSteps - SUCCESS_THRESHOLD_MARGIN,
        ).length;

        console.log(
            `\n📊 Статистика после ${this.totalEpisodes} эпизодов:\n` +
            `   ✅ Успехов: ${this.totalSuccesses} (${successRate}%)\n` +
            `   📈 Подряд успехов: ${this.successStreak}\n` +
            `   📉 Подряд неудач: ${this.failureStreak}\n` +
            `   🎯 CurrentMaxSteps: ${this.currentMaxSteps}\n` +
            `   📏 Средняя длина (20 эп.): ${avgSteps} шагов\n` +
            `   🔥 Успехов из последних 10: ${recentSuccesses}`,
        );
    }

    /**
     * Получить текущее значение maxSteps (для UI).
     * @returns {number}
     */
    getCurrentMaxSteps() {
        return this.currentMaxSteps;
    }

    /**
     * Получить статистику для UI.
     * @returns {Object}
     */
    getAdaptiveStats() {
        const successRate = this.totalEpisodes > 0
            ? ((this.totalSuccesses / this.totalEpisodes) * 100).toFixed(1)
            : '0.0';

        const avgSteps = this.episodeStepsHistory.length > 0
            ? (this.episodeStepsHistory.reduce((a, b) => a + b, 0) / this.episodeStepsHistory.length).toFixed(0)
            : '0';

        return {
            totalEpisodes:   this.totalEpisodes,
            totalSuccesses:  this.totalSuccesses,
            successRate:     successRate,
            successStreak:   this.successStreak,
            failureStreak:   this.failureStreak,
            currentMaxSteps: this.currentMaxSteps,
            avgSteps:        avgSteps,
        };
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
     * Переопределяем move() для трекинга глобальной памяти.
     * @param {number} actionIndex
     */
    move(actionIndex) {
        const prevPos = { ...this.pos };

        super.move(actionIndex);

        // Обновить глобальную память только если движение было успешным
        if (this.pos.x !== prevPos.x || this.pos.y !== prevPos.y) {
            this.updateGlobalMemory(this.pos.x, this.pos.y);
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
