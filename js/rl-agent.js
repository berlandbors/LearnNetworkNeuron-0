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

// ── Константы обнаружения застревания ────────────────────────────────────────

/** Окно последних шагов для анализа зацикливания */
const STUCK_DETECTION_WINDOW = 20;
/** Максимальное количество уникальных клеток в окне, считающееся зацикливанием */
const STUCK_LOOP_CELLS = 4;
/** Минимум уникальных клеток за весь эпизод (ниже = подозрение на застревание) */
const STUCK_MIN_UNIQUE_CELLS = 15;
/** Максимальная доля повторных посещений (выше = застревание) */
const STUCK_MAX_REPEAT_RATE = 0.6;

// ── Константы адаптивного maxSteps ───────────────────────────────────────────

/** Интервал успехов для уменьшения maxSteps */
const SUCCESS_REDUCTION_INTERVAL = 5;
/** Количество неудач подряд для увеличения maxSteps */
const FAILURE_INCREASE_THRESHOLD = 10;
/** Максимальное количество хранимых причин завершения эпизодов */
const MAX_EPISODE_HISTORY = 100;

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

        /** Начальный лимит шагов (большой — для исследования на старте) */
        this.baseMaxSteps = 3000;
        /** Минимальный лимит шагов (для обученного агента) */
        this.minMaxSteps = 300;
        /** Текущий лимит шагов (изменяется динамически) */
        this.currentMaxSteps = this.baseMaxSteps;

        /** Счётчик успешных эпизодов */
        this.successCount = 0;
        /** Количество неудач подряд (без успеха) */
        this.consecutiveFailures = 0;
        /** Общее количество эпизодов */
        this.totalEpisodes = 0;

        /** История изменений currentMaxSteps */
        this.maxStepsHistory = [];
        /** Причины завершения последних эпизодов */
        this.episodeReasons = [];
    }

    // ── Обнаружение застревания и управление эпизодом ────────────────────────

    /**
     * Определить, застрял ли агент (зацикливание).
     * @returns {boolean}
     */
    isStuck() {
        if (this.steps < 100) return false;

        const uniqueCells  = this.visited.size;
        const totalSteps   = this.steps;
        const repeatRate   = (totalSteps - uniqueCells) / totalSteps;

        // Явное зацикливание: последние N шагов — только 3-4 клетки
        const recentPathSize = STUCK_DETECTION_WINDOW;
        if (this.path.length >= recentPathSize) {
            const recentCells = new Set();
            for (let i = this.path.length - recentPathSize; i < this.path.length; i++) {
                recentCells.add(`${this.path[i].x},${this.path[i].y}`);
            }
            if (recentCells.size <= STUCK_LOOP_CELLS) return true;
        }

        // Слишком мало уникальных клеток и высокий процент повторов
        return uniqueCells < STUCK_MIN_UNIQUE_CELLS && repeatRate > STUCK_MAX_REPEAT_RATE;
    }

    /**
     * Получить текст причины застревания (для логирования).
     * @returns {string}
     */
    getStuckReason() {
        const uniqueCells = this.visited.size;
        const repeatRate  = ((this.steps - uniqueCells) / this.steps * 100).toFixed(1);
        return `${uniqueCells} клеток, ${repeatRate}% повторов`;
    }

    /**
     * Проверить, завершён ли эпизод.
     * @returns {boolean}
     */
    isDone() {
        return this.reached ||
               this.isStuck() ||
               this.steps >= this.currentMaxSteps;
    }

    /**
     * Получить причину завершения эпизода (для UI и логов).
     * @returns {{reason: string, icon: string, color: string}}
     */
    getEpisodeEndReason() {
        if (this.reached) {
            return {
                reason: `Цель достигнута за ${this.steps} шагов`,
                icon:   '🎯',
                color:  '#10b981',
            };
        } else if (this.isStuck()) {
            return {
                reason: `Застрял на ${this.steps} шагах (${this.getStuckReason()})`,
                icon:   '🔄',
                color:  '#f59e0b',
            };
        } else {
            return {
                reason: `Лимит шагов (${this.currentMaxSteps})`,
                icon:   '⏱️',
                color:  '#6366f1',
            };
        }
    }

    /**
     * Обновить currentMaxSteps на основе результатов завершённого эпизода.
     * Должен вызываться в начале reset() (до super.reset, чтобы this.reached актуален).
     */
    updateMaxSteps() {
        const prevMaxSteps = this.currentMaxSteps;

        if (this.reached) {
            // ✅ Успех — уменьшаем лимит
            this.successCount++;
            this.consecutiveFailures = 0;

            // Уменьшаем каждые SUCCESS_REDUCTION_INTERVAL успехов
            if (this.successCount % SUCCESS_REDUCTION_INTERVAL === 0) {
                this.currentMaxSteps = Math.max(
                    this.minMaxSteps,
                    Math.floor(this.currentMaxSteps * 0.85),
                );
            }

            // Более агрессивное уменьшение при стабильных успехах
            if (this.successCount >= 20 && this.currentMaxSteps > this.minMaxSteps * 1.5) {
                this.currentMaxSteps = Math.max(
                    this.minMaxSteps,
                    Math.floor(this.currentMaxSteps * 0.9),
                );
            }

        } else if (this.isStuck()) {
            // 🔄 Застрял — мягкое уменьшение (не тратить время на долгие циклы)
            this.currentMaxSteps = Math.max(
                this.minMaxSteps,
                Math.floor(this.currentMaxSteps * 0.95),
            );

        } else {
            // ❌ Таймаут — увеличиваем лимит после 10 неудач подряд
            this.consecutiveFailures++;

            if (this.consecutiveFailures >= FAILURE_INCREASE_THRESHOLD) {
                this.currentMaxSteps = Math.min(
                    this.baseMaxSteps,
                    Math.floor(this.currentMaxSteps * 1.2),
                );
                this.consecutiveFailures = 0;
                console.warn(`⚠️ MaxSteps увеличен до ${this.currentMaxSteps} (10 неудач подряд)`);
            }
        }

        // Логирование изменений
        if (prevMaxSteps !== this.currentMaxSteps) {
            this.maxStepsHistory.push({
                episode:  this.totalEpisodes,
                oldValue: prevMaxSteps,
                newValue: this.currentMaxSteps,
                reason:   this.reached ? 'success' : (this.isStuck() ? 'stuck' : 'failures'),
            });
            console.log(`📊 MaxSteps: ${prevMaxSteps} → ${this.currentMaxSteps}`);
        }
    }

    // ── Глобальная память ─────────────────────────────────────────────────────

    /**
     * Сбросить состояние эпизода, НО сохранить глобальную память.
     * @param {{x:number,y:number}} startPos
     */
    reset(startPos) {
        // 1. Сохраняем глобальную память и статистику ДО вызова parent.reset()
        const globalVisited    = this.globalVisited;
        const globalVisitCount = this.globalVisitCount;
        const episodeCount     = this.episodeCount;

        // 2. Сохраняем причину завершения эпизода (перед super.reset сбрасывает состояние)
        // steps > 0 исключает первый вызов из setEnvironment (инициализация)
        if (this.steps > 0) {
            this.episodeReasons.push(this.getEpisodeEndReason());
            if (this.episodeReasons.length > MAX_EPISODE_HISTORY) {
                this.episodeReasons.shift();
            }
        }

        // 3. Адаптируем maxSteps на основе результатов завершённого эпизода
        // steps > 0 исключает первый вызов из setEnvironment (инициализация)
        if (this.steps > 0) {
            this.updateMaxSteps();
        }

        // 4. Вызываем родительский reset (стирает локальную память)
        super.reset(startPos);

        // 5. Восстанавливаем глобальную память
        this.globalVisited    = globalVisited;
        this.globalVisitCount = globalVisitCount;
        this.episodeCount     = episodeCount + 1;
        this.totalEpisodes++;

        // 6. Затухание памяти каждые 50 эпизодов
        if (this.episodeCount % 50 === 0) {
            this.decayGlobalMemory(0.7);
        }

        // 7. Обнулить награду эпизода
        this.episodeReward = 0;
    }

    // ── Статистика адаптивного maxSteps ──────────────────────────────────────

    /**
     * Получить статистику адаптации maxSteps.
     * @returns {{current: number, min: number, max: number, changes: number, successRate: string}}
     */
    getMaxStepsStats() {
        return {
            current:     this.currentMaxSteps,
            min:         this.minMaxSteps,
            max:         this.baseMaxSteps,
            changes:     this.maxStepsHistory.length,
            successRate: this.totalEpisodes > 0
                ? (this.successCount / this.totalEpisodes * 100).toFixed(1)
                : '0',
        };
    }

    /**
     * Получить статистику причин завершения за последние N эпизодов.
     * @param {number} [n=20]
     * @returns {{goal: number, stuck: number, timeout: number}}
     */
    getEpisodeEndStats(n = 20) {
        const recent = this.episodeReasons.slice(-n);
        return {
            goal:    recent.filter(r => r.icon === '🎯').length,
            stuck:   recent.filter(r => r.icon === '🔄').length,
            timeout: recent.filter(r => r.icon === '⏱️').length,
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
     * Переопределяем move() чтобы обновлять глобальную память.
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
