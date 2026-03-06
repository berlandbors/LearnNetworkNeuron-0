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
