/**
 * rl-agent.js — агент с Q-обучением (Deep Q-Network, упрощённый вариант).
 *
 * Использует:
 *  - epsilon-greedy стратегию исследования
 *  - experience replay с мини-батчами
 *  - отдельную целевую сеть (target network) для стабильности
 */
import { Agent, INPUT_SIZE, OUTPUT_SIZE } from './agent.js';
import { NeuralNetwork } from './neural-network.js';

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
/** Скорость обучения */
const LEARNING_RATE = 0.001;
/** Шаг обновления целевой сети */
const TARGET_UPDATE_FREQ = 100;

export class RLAgent extends Agent {
    /**
     * @param {number} hiddenSize - нейронов в скрытом слое
     */
    constructor(hiddenSize = 16) {
        super(null, hiddenSize);

        /** Основная Q-сеть */
        this.brain = new NeuralNetwork(INPUT_SIZE, hiddenSize, OUTPUT_SIZE);

        /** Целевая Q-сеть (обновляется реже) */
        this.targetBrain = this.brain.copy();

        /** Буфер опыта */
        this.memory = [];

        /** Счётчик шагов (для обновления целевой сети) */
        this.totalSteps = 0;

        /** Текущий epsilon */
        this.epsilon = EPSILON_START;

        /** Суммарные награды за эпизод */
        this.episodeReward = 0;

        /** История наград по эпизодам */
        this.rewardHistory = [];
    }

    /**
     * Сохранить опыт в буфер.
     * @param {number[]} state      - состояние до действия
     * @param {number}   action     - выполненное действие
     * @param {number}   reward     - полученная награда
     * @param {number[]} nextState  - состояние после действия
     * @param {boolean}  done       - эпизод завершён
     */
    remember(state, action, reward, nextState, done) {
        this.memory.push({ state, action, reward, nextState, done });
        if (this.memory.length > MEMORY_SIZE) {
            this.memory.shift();
        }
        this.episodeReward += reward;
    }

    /**
     * Выбрать действие по epsilon-greedy стратегии.
     * @param {number[]} state - текущее состояние
     * @returns {number} индекс действия
     */
    act(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * OUTPUT_SIZE);
        }
        const qValues = this.brain.predict(state);
        return qValues.indexOf(Math.max(...qValues));
    }

    /**
     * Обучение на мини-батче из буфера памяти (Experience Replay).
     * Упрощённый градиентный спуск через конечные разности.
     * @param {number} [batchSize=32]
     */
    replay(batchSize = 32) {
        if (this.memory.length < batchSize) return;

        // Случайный мини-батч
        const batch = [];
        const indices = new Set();
        while (indices.size < batchSize) {
            indices.add(Math.floor(Math.random() * this.memory.length));
        }
        for (const i of indices) batch.push(this.memory[i]);

        for (const { state, action, reward, nextState, done } of batch) {
            // Вычислить целевое Q-значение
            const target = done
                ? reward
                : reward + GAMMA * Math.max(...this.targetBrain.predict(nextState));

            // Обновить вес действия через конечные разности (упрощённый SGD)
            this._updateWeights(state, action, target);
        }

        // Обновить целевую сеть с заданной периодичностью
        this.totalSteps++;
        if (this.totalSteps % TARGET_UPDATE_FREQ === 0) {
            this.targetBrain = this.brain.copy();
        }
    }

    /**
     * Упрощённое обновление весов: подталкиваем веса в направлении
     * уменьшения ошибки через численный градиент.
     * @param {number[]} state
     * @param {number}   action
     * @param {number}   target
     */
    _updateWeights(state, action, target) {
        const qValues = this.brain.predict(state);
        const error = target - qValues[action];

        // Обновляем только выходные веса, связанные с выбранным действием
        // (упрощённый, но рабочий подход для демонстрации)
        const delta = LEARNING_RATE * error;
        const hiddenSize = this.brain.hiddenSize;

        // Пересчитать скрытый слой для нахождения активаций
        const hidden = this.brain.biasH.map((b, i) => {
            const sum = this.brain.weightsIH[i].reduce(
                (acc, w, j) => acc + w * (state[j] ?? 0), b
            );
            return Math.max(0, sum); // ReLU
        });

        // Обновить выходные веса для выбранного действия
        for (let j = 0; j < hiddenSize; j++) {
            this.brain.weightsHO[action][j] += delta * hidden[j];
        }
        this.brain.biasO[action] += delta;
    }

    /**
     * Уменьшить epsilon (снизить долю случайных действий).
     */
    updateEpsilon() {
        this.rewardHistory.push(this.episodeReward);
        this.episodeReward = 0;

        if (this.epsilon > EPSILON_MIN) {
            this.epsilon *= EPSILON_DECAY;
        }
    }
}
