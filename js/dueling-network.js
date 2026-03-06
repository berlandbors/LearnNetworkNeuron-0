/**
 * dueling-network.js — Dueling DQN архитектура.
 *
 * Разделяет последний скрытый слой на два потока:
 *  - Value stream  V(s)    — оценка ценности состояния
 *  - Advantage stream A(s,a) — преимущество каждого действия
 *
 * Итоговое Q-значение: Q(s,a) = V(s) + (A(s,a) − mean(A(s,a)))
 *
 * Источник: Wang et al., 2016 — "Dueling Network Architectures for DRL"
 */
import { AdvancedNeuralNetwork } from './neural-network-advanced.js';

export class DuelingNetwork extends AdvancedNeuralNetwork {
    /**
     * @param {number}   inputSize    - число входных нейронов
     * @param {number[]} hiddenLayers - массив скрытых слоёв, напр. [64, 32]
     * @param {number}   outputSize   - число действий
     * @param {object}   [options]    - те же опции, что у AdvancedNeuralNetwork
     */
    constructor(inputSize, hiddenLayers, outputSize, options = {}) {
        super(inputSize, hiddenLayers, outputSize, options);

        const commonSize = hiddenLayers[hiddenLayers.length - 1];

        // Xavier инициализация для двух дополнительных потоков
        const scaleV = Math.sqrt(2 / commonSize);
        const scaleA = Math.sqrt(2 / commonSize);

        // Value stream: [commonSize] → [1]
        this.valueWeights = [
            Array.from({ length: commonSize }, () => (Math.random() * 2 - 1) * scaleV)
        ];
        this.valueBias = [0];

        // Advantage stream: [commonSize] → [outputSize]
        this.advantageWeights = Array.from({ length: outputSize }, () =>
            Array.from({ length: commonSize }, () => (Math.random() * 2 - 1) * scaleA)
        );
        this.advantageBias = new Array(outputSize).fill(0);

        // Adam-моменты для потоков
        this._initDuelingAdam(commonSize, outputSize);
    }

    _initDuelingAdam(commonSize, outputSize) {
        this.mvW = [new Array(commonSize).fill(0)];
        this.vvW = [new Array(commonSize).fill(0)];
        this.mvb = [0];
        this.vvb = [0];

        this.maW = Array.from({ length: outputSize }, () => new Array(commonSize).fill(0));
        this.vaW = Array.from({ length: outputSize }, () => new Array(commonSize).fill(0));
        this.mab = new Array(outputSize).fill(0);
        this.vab = new Array(outputSize).fill(0);
    }

    /**
     * Прямой проход до последнего скрытого слоя (общая часть).
     * @param {number[]} inputs
     * @returns {number[]} активации последнего скрытого слоя
     */
    _forwardToLastHidden(inputs) {
        let a = inputs.slice();
        const L = this.W.length - 1; // без выходного слоя

        for (let l = 0; l < L; l++) {
            a = this._linearLayer(a, this.W[l], this.b[l]);
            if (this.useBatchNorm && this.batchNormLayers) {
                a = this.batchNormLayers[l].forward(a, false);
            }
            a = a.map(x => Math.max(0, x)); // ReLU
        }

        return a;
    }

    /**
     * Предсказание Q-значений через Dueling потоки.
     * @param {number[]} inputs
     * @returns {number[]}
     */
    predict(inputs) {
        const hidden = this._forwardToLastHidden(inputs);

        // Value stream (linear activation for expressiveness)
        const value = this.valueBias[0] +
            this.valueWeights[0].reduce((sum, w, j) => sum + w * hidden[j], 0);
        const V = value;

        // Advantage stream (linear activation)
        const advantages = this.advantageBias.map((b, i) => {
            return b + this.advantageWeights[i].reduce((sum, w, j) => sum + w * hidden[j], 0);
        });

        const meanA = advantages.reduce((a, b) => a + b, 0) / advantages.length;

        // Q(s,a) = V(s) + A(s,a) − mean(A)
        return advantages.map(a => V + (a - meanA));
    }

    /**
     * Обучение на одном примере (переопределяем для учёта двойных потоков).
     * @param {number[]} state
     * @param {number}   action
     * @param {number}   target
     * @param {number}   [isWeight=1]
     */
    trainOnSample(state, action, target, isWeight = 1) {
        // Используем базовую backprop для общих слоёв,
        // затем дополнительно обновляем Dueling потоки через численный градиент.
        super.trainOnSample(state, action, target, isWeight);
        this._updateDuelingStreams(state, action, target, isWeight);
    }

    /**
     * Обновить веса Dueling потоков через численный градиент.
     */
    _updateDuelingStreams(state, action, target, isWeight) {
        const hidden   = this._forwardToLastHidden(state);
        const qValues  = this.predict(state);
        const error    = isWeight * (target - qValues[action]);

        // Используем счётчик итерации из базового класса (уже увеличен в backward())
        const t  = this.iteration;
        const lr = this.learningRate;
        const b1 = this.beta1;
        const b2 = this.beta2;
        const ep = this.adamEps;

        // Обновить value stream
        for (let k = 0; k < hidden.length; k++) {
            const g = -error * hidden[k];
            this.mvW[0][k] = b1 * this.mvW[0][k] + (1 - b1) * g;
            this.vvW[0][k] = b2 * this.vvW[0][k] + (1 - b2) * g * g;
            const mH = this.mvW[0][k] / (1 - b1 ** t);
            const vH = this.vvW[0][k] / (1 - b2 ** t);
            this.valueWeights[0][k] -= lr * mH / (Math.sqrt(vH) + ep);
        }
        const gb = -error;
        this.mvb[0] = b1 * this.mvb[0] + (1 - b1) * gb;
        this.vvb[0] = b2 * this.vvb[0] + (1 - b2) * gb * gb;
        this.valueBias[0] -= lr * (this.mvb[0] / (1 - b1 ** t)) /
            (Math.sqrt(this.vvb[0] / (1 - b2 ** t)) + ep);

        // Обновить advantage stream для выбранного действия
        for (let k = 0; k < hidden.length; k++) {
            const g = -error * hidden[k];
            this.maW[action][k] = b1 * this.maW[action][k] + (1 - b1) * g;
            this.vaW[action][k] = b2 * this.vaW[action][k] + (1 - b2) * g * g;
            const mH = this.maW[action][k] / (1 - b1 ** t);
            const vH = this.vaW[action][k] / (1 - b2 ** t);
            this.advantageWeights[action][k] -= lr * mH / (Math.sqrt(vH) + ep);
        }
        this.mab[action] = b1 * this.mab[action] + (1 - b1) * gb;
        this.vab[action] = b2 * this.vab[action] + (1 - b2) * gb * gb;
        this.advantageBias[action] -= lr * (this.mab[action] / (1 - b1 ** t)) /
            (Math.sqrt(this.vab[action] / (1 - b2 ** t)) + ep);
    }

    /** Создать глубокую копию. */
    copy() {
        const clone = new DuelingNetwork(
            this.layerSizes[0],
            this.layerSizes.slice(1, -1),
            this.layerSizes[this.layerSizes.length - 1],
            {
                learningRate: this.learningRate,
                dropoutRate:  this.dropoutRate,
                useBatchNorm: this.useBatchNorm,
            }
        );

        clone.W  = this.W.map(l => l.map(r => r.slice()));
        clone.b  = this.b.map(l => l.slice());
        clone.mW = this.mW.map(l => l.map(r => r.slice()));
        clone.vW = this.vW.map(l => l.map(r => r.slice()));
        clone.mb = this.mb.map(l => l.slice());
        clone.vb = this.vb.map(l => l.slice());
        clone.iteration = this.iteration;

        clone.valueWeights     = this.valueWeights.map(r => r.slice());
        clone.valueBias        = this.valueBias.slice();
        clone.advantageWeights = this.advantageWeights.map(r => r.slice());
        clone.advantageBias    = this.advantageBias.slice();
        clone._initDuelingAdam(
            this.layerSizes[this.layerSizes.length - 2],
            this.layerSizes[this.layerSizes.length - 1]
        );
        clone._syncLegacyWeights();

        return clone;
    }

    /** Сериализовать. */
    serialize() {
        return {
            ...super.serialize(),
            type:             'DuelingNetwork',
            valueWeights:     this.valueWeights,
            valueBias:        this.valueBias,
            advantageWeights: this.advantageWeights,
            advantageBias:    this.advantageBias,
        };
    }

    /**
     * Восстановить из данных.
     * @param {object} data
     * @returns {DuelingNetwork}
     */
    static deserialize(data) {
        const net = new DuelingNetwork(
            data.layerSizes[0],
            data.layerSizes.slice(1, -1),
            data.layerSizes[data.layerSizes.length - 1],
            {
                learningRate: data.learningRate ?? 0.001,
                dropoutRate:  data.dropoutRate  ?? 0.0,
                useBatchNorm: data.useBatchNorm ?? false,
            }
        );
        net.W                = data.W;
        net.b                = data.b;
        net.valueWeights     = data.valueWeights;
        net.valueBias        = data.valueBias;
        net.advantageWeights = data.advantageWeights;
        net.advantageBias    = data.advantageBias;
        net._syncLegacyWeights();
        return net;
    }
}
