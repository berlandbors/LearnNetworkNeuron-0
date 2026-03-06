/**
 * neural-network-advanced.js — расширенная нейронная сеть с:
 *  - Xavier/He инициализацией весов
 *  - Backpropagation и Adam optimizer
 *  - Batch Normalization
 *  - Dropout для регуляризации
 *
 * Поддерживает произвольное число скрытых слоёв.
 */
import { NeuralNetwork } from './neural-network.js';

// ─── Batch Normalization ──────────────────────────────────────────────────────

/**
 * Слой нормализации по батчу (работает с одним вектором активаций).
 */
export class BatchNormLayer {
    /**
     * @param {number} size - число нейронов в слое
     */
    constructor(size) {
        this.size = size;
        /** Обучаемый масштаб */
        this.gamma = new Array(size).fill(1);
        /** Обучаемый сдвиг */
        this.beta = new Array(size).fill(0);
        /** Скользящее среднее (для инференса) */
        this.runningMean = new Array(size).fill(0);
        /** Скользящая дисперсия (для инференса) */
        this.runningVar = new Array(size).fill(1);
        this.momentum = 0.9;
        this.epsilon = 1e-5;
    }

    /**
     * Прямой проход.
     * @param {number[]} inputs
     * @param {boolean} training - true во время обучения
     * @returns {number[]}
     */
    forward(inputs, training = true) {
        if (training) {
            const n = inputs.length;
            const mean = inputs.reduce((a, b) => a + b, 0) / n;
            const variance = inputs.reduce((a, x) => a + (x - mean) ** 2, 0) / n;

            // Обновить per-neuron running statistics инкрементально
            this.runningMean = this.runningMean.map((m, i) =>
                this.momentum * m + (1 - this.momentum) * inputs[i]
            );
            this.runningVar = this.runningVar.map((v, i) => {
                const diff = inputs[i] - this.runningMean[i];
                return this.momentum * v + (1 - this.momentum) * diff * diff;
            });

            // Нормализовать относительно батч-статистики
            return inputs.map((x, i) =>
                this.gamma[i] * (x - mean) / Math.sqrt(variance + this.epsilon) + this.beta[i]
            );
        } else {
            return inputs.map((x, i) =>
                this.gamma[i] * (x - this.runningMean[i]) /
                Math.sqrt(this.runningVar[i] + this.epsilon) + this.beta[i]
            );
        }
    }
}

// ─── AdvancedNeuralNetwork ────────────────────────────────────────────────────

/**
 * Многослойная нейронная сеть с backpropagation и Adam optimizer.
 * Расширяет базовый NeuralNetwork для сохранения совместимости
 * с генетическим алгоритмом (copy, mutate, crossover, serialize).
 */
export class AdvancedNeuralNetwork extends NeuralNetwork {
    /**
     * @param {number}   inputSize   - число входных нейронов
     * @param {number[]} hiddenLayers - массив размеров скрытых слоёв, напр. [64, 32]
     * @param {number}   outputSize  - число выходных нейронов
     * @param {object}   [options]
     * @param {number}   [options.learningRate=0.001]
     * @param {number}   [options.dropoutRate=0.0]  - доля выключаемых нейронов (0 = выкл.)
     * @param {boolean}  [options.useBatchNorm=false]
     */
    constructor(inputSize, hiddenLayers, outputSize, options = {}) {
        // Вызываем конструктор базового класса с первым скрытым слоем
        super(inputSize, hiddenLayers[0], outputSize);

        this.layerSizes = [inputSize, ...hiddenLayers, outputSize];

        const {
            learningRate = 0.001,
            dropoutRate  = 0.0,
            useBatchNorm = false,
        } = options;

        this.learningRate = learningRate;
        this.dropoutRate  = dropoutRate;
        this.useBatchNorm = useBatchNorm;

        // Adam гиперпараметры
        this.beta1   = 0.9;
        this.beta2   = 0.999;
        this.adamEps = 1e-8;
        this.iteration = 0;

        // Переинициализируем все слои (заменяем двухслойную структуру базового класса)
        this._initLayers();

        if (useBatchNorm) {
            this._initBatchNorm();
        }
    }

    // ── Инициализация ──────────────────────────────────────────────────────────

    /**
     * He/Xavier инициализация весов для всех слоёв.
     */
    _initLayers() {
        const L = this.layerSizes.length - 1; // число матриц весов
        this.W  = [];   // W[l]: матрица весов слоя l
        this.b  = [];   // b[l]: смещения слоя l
        this.mW = [];   // Adam: первый момент для W
        this.vW = [];   // Adam: второй момент для W
        this.mb = [];   // Adam: первый момент для b
        this.vb = [];   // Adam: второй момент для b

        for (let l = 0; l < L; l++) {
            const fanIn  = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            // He initialization (для ReLU)
            const scale = Math.sqrt(2 / fanIn);

            this.W[l]  = Array.from({ length: fanOut }, () =>
                Array.from({ length: fanIn }, () => (Math.random() * 2 - 1) * scale)
            );
            this.b[l]  = new Array(fanOut).fill(0);
            this.mW[l] = Array.from({ length: fanOut }, () => new Array(fanIn).fill(0));
            this.vW[l] = Array.from({ length: fanOut }, () => new Array(fanIn).fill(0));
            this.mb[l] = new Array(fanOut).fill(0);
            this.vb[l] = new Array(fanOut).fill(0);
        }

        // Для совместимости с базовым классом (weightsIH, weightsHO, biasH, biasO)
        this._syncLegacyWeights();
    }

    /** Инициализировать Batch Norm слои для всех скрытых слоёв */
    _initBatchNorm() {
        this.batchNormLayers = this.layerSizes.slice(1, -1).map(size => new BatchNormLayer(size));
    }

    /**
     * Синхронизировать устаревшие поля (weightsIH и т.д.) с новыми W/b.
     * Нужно для совместимости с базовым NeuralNetwork (copy, serialize…).
     */
    _syncLegacyWeights() {
        if (this.W.length >= 1) {
            this.weightsIH = this.W[0];
            this.biasH     = this.b[0];
        }
        if (this.W.length >= 2) {
            this.weightsHO = this.W[this.W.length - 1];
            this.biasO     = this.b[this.b.length - 1];
        }
    }

    // ── Прямой проход ──────────────────────────────────────────────────────────

    /**
     * Прямой проход по всем слоям (ReLU на скрытых, Sigmoid на выходе).
     * @param {number[]} inputs
     * @param {boolean}  [training=false] - включить dropout и batch norm в режиме обучения
     * @returns {number[]}
     */
    predict(inputs, training = false) {
        let a = inputs.slice();
        const L = this.W.length;

        for (let l = 0; l < L; l++) {
            a = this._linearLayer(a, this.W[l], this.b[l]);

            if (l < L - 1) {
                // Скрытый слой: Batch Norm → ReLU → Dropout
                if (this.useBatchNorm && this.batchNormLayers) {
                    a = this.batchNormLayers[l].forward(a, training);
                }
                a = a.map(x => Math.max(0, x)); // ReLU
                if (training && this.dropoutRate > 0) {
                    a = this._applyDropout(a, this.dropoutRate);
                }
            } else {
                // Выходной слой: Sigmoid
                a = a.map(x => 1 / (1 + Math.exp(-x)));
            }
        }

        return a;
    }

    /**
     * Прямой проход с сохранением промежуточных активаций (нужно для backprop).
     * @param {number[]} inputs
     * @returns {{ zs: number[][], as: number[][] }}
     *   zs[l] — линейный вход в слой l (до активации)
     *   as[l] — активации после слоя l (as[0] == inputs)
     */
    _forwardWithCache(inputs) {
        const zs = [];
        const as = [inputs.slice()];
        const L  = this.W.length;

        for (let l = 0; l < L; l++) {
            const z = this._linearLayer(as[l], this.W[l], this.b[l]);
            zs.push(z);

            let a;
            if (l < L - 1) {
                let bn = z;
                if (this.useBatchNorm && this.batchNormLayers) {
                    bn = this.batchNormLayers[l].forward(z, true);
                }
                a = bn.map(x => Math.max(0, x)); // ReLU
            } else {
                a = z.map(x => 1 / (1 + Math.exp(-x))); // Sigmoid
            }
            as.push(a);
        }

        return { zs, as };
    }

    // ── Backpropagation ────────────────────────────────────────────────────────

    /**
     * Обратное распространение ошибки для одного примера.
     * @param {number[]} inputs   - входной вектор
     * @param {number[]} targets  - целевой вектор Q-значений
     * @param {number}   [isWeight=1] - вес примера (importance sampling)
     */
    backward(inputs, targets, isWeight = 1) {
        this.iteration++;

        const { zs, as } = this._forwardWithCache(inputs);
        const L = this.W.length;

        // Градиент ошибки MSE на выходном слое
        // δ[L] = (output - target) * sigmoid'(z[L])
        const output = as[L];
        let delta = output.map((o, i) => {
            const sigPrime = o * (1 - o); // производная sigmoid
            return isWeight * (o - targets[i]) * sigPrime;
        });

        const gradW = this.W.map(layer => layer.map(row => new Array(row.length).fill(0)));
        const gradb = this.b.map(layer => new Array(layer.length).fill(0));

        for (let l = L - 1; l >= 0; l--) {
            // Накопить градиенты для W[l] и b[l]
            for (let j = 0; j < this.W[l].length; j++) {
                gradb[l][j] = delta[j];
                for (let k = 0; k < this.W[l][j].length; k++) {
                    gradW[l][j][k] = delta[j] * as[l][k];
                }
            }

            if (l > 0) {
                // Передать дельту на предыдущий слой
                const prevDelta = new Array(as[l].length).fill(0);
                for (let k = 0; k < as[l].length; k++) {
                    let s = 0;
                    for (let j = 0; j < this.W[l].length; j++) {
                        s += this.W[l][j][k] * delta[j];
                    }
                    // ReLU производная: 1 если z > 0
                    prevDelta[k] = s * (zs[l - 1][k] > 0 ? 1 : 0);
                }
                delta = prevDelta;
            }
        }

        // Adam update
        this._adamUpdate(gradW, gradb);
        this._syncLegacyWeights();
    }

    /**
     * Обучение на одном примере с заданным целевым Q-значением для одного действия.
     * @param {number[]} state  - входной вектор
     * @param {number}   action - индекс действия
     * @param {number}   target - целевое Q-значение
     * @param {number}   [isWeight=1] - importance sampling weight
     */
    trainOnSample(state, action, target, isWeight = 1) {
        const qValues = this.predict(state, true);
        const targets = qValues.slice(); // копия
        targets[action] = target;
        this.backward(state, targets, isWeight);
    }

    /**
     * Обучение на мини-батче.
     * @param {{ state: number[], action: number, reward: number,
     *           nextState: number[], done: boolean,
     *           weight?: number }[]} batch
     * @param {NeuralNetwork} targetNet - целевая сеть для Double DQN
     * @param {number} gamma - коэффициент дисконтирования
     */
    train(batch, targetNet, gamma = 0.95) {
        for (const { state, action, reward, nextState, done, weight = 1 } of batch) {
            const qNext   = Math.max(...targetNet.predict(nextState));
            const target  = done ? reward : reward + gamma * qNext;
            this.trainOnSample(state, action, target, weight);
        }
    }

    // ── Adam ───────────────────────────────────────────────────────────────────

    /**
     * Обновить веса методом Adam.
     * @param {number[][][]} gradW
     * @param {number[][]}   gradb
     */
    _adamUpdate(gradW, gradb) {
        const t  = this.iteration;
        const lr = this.learningRate;
        const b1 = this.beta1;
        const b2 = this.beta2;
        const ep = this.adamEps;

        for (let l = 0; l < this.W.length; l++) {
            for (let j = 0; j < this.W[l].length; j++) {
                for (let k = 0; k < this.W[l][j].length; k++) {
                    const g = gradW[l][j][k];
                    this.mW[l][j][k] = b1 * this.mW[l][j][k] + (1 - b1) * g;
                    this.vW[l][j][k] = b2 * this.vW[l][j][k] + (1 - b2) * g * g;
                    const mHat = this.mW[l][j][k] / (1 - b1 ** t);
                    const vHat = this.vW[l][j][k] / (1 - b2 ** t);
                    this.W[l][j][k] -= lr * mHat / (Math.sqrt(vHat) + ep);
                }
                const gb = gradb[l][j];
                this.mb[l][j] = b1 * this.mb[l][j] + (1 - b1) * gb;
                this.vb[l][j] = b2 * this.vb[l][j] + (1 - b2) * gb * gb;
                const mbHat = this.mb[l][j] / (1 - b1 ** t);
                const vbHat = this.vb[l][j] / (1 - b2 ** t);
                this.b[l][j] -= lr * mbHat / (Math.sqrt(vbHat) + ep);
            }
        }
    }

    // ── Вспомогательные ────────────────────────────────────────────────────────

    /**
     * Линейное преобразование: z = W·a + b
     * @param {number[]} a  - вектор входов
     * @param {number[][]} W - матрица весов [out × in]
     * @param {number[]} b  - смещения
     * @returns {number[]}
     */
    _linearLayer(a, W, b) {
        return W.map((row, j) =>
            row.reduce((sum, w, k) => sum + w * (a[k] ?? 0), b[j])
        );
    }

    /**
     * Dropout: обнулить случайные активации и масштабировать остальные.
     * @param {number[]} a
     * @param {number}   rate
     * @returns {number[]}
     */
    _applyDropout(a, rate) {
        const scale = 1 / (1 - rate);
        return a.map(x => (Math.random() < rate ? 0 : x * scale));
    }

    // ── Копирование / сериализация ─────────────────────────────────────────────

    /** Создать глубокую копию сети. */
    copy() {
        const clone = new AdvancedNeuralNetwork(
            this.layerSizes[0],
            this.layerSizes.slice(1, -1),
            this.layerSizes[this.layerSizes.length - 1],
            {
                learningRate: this.learningRate,
                dropoutRate:  this.dropoutRate,
                useBatchNorm: this.useBatchNorm,
            }
        );

        clone.W  = this.W.map(layer => layer.map(row => row.slice()));
        clone.b  = this.b.map(layer => layer.slice());
        clone.mW = this.mW.map(layer => layer.map(row => row.slice()));
        clone.vW = this.vW.map(layer => layer.map(row => row.slice()));
        clone.mb = this.mb.map(layer => layer.slice());
        clone.vb = this.vb.map(layer => layer.slice());
        clone.iteration = this.iteration;
        clone._syncLegacyWeights();

        return clone;
    }

    /** Сериализовать в JSON-объект. */
    serialize() {
        return {
            type:        'AdvancedNeuralNetwork',
            layerSizes:  this.layerSizes,
            learningRate: this.learningRate,
            dropoutRate:  this.dropoutRate,
            useBatchNorm: this.useBatchNorm,
            W:           this.W,
            b:           this.b,
            // Поля для совместимости с базовым deserialize
            inputSize:   this.inputSize,
            hiddenSize:  this.hiddenSize,
            outputSize:  this.outputSize,
            weightsIH:   this.weightsIH,
            weightsHO:   this.weightsHO,
            biasH:       this.biasH,
            biasO:       this.biasO,
        };
    }

    /**
     * Восстановить из сериализованных данных.
     * @param {object} data
     * @returns {AdvancedNeuralNetwork}
     */
    static deserialize(data) {
        const net = new AdvancedNeuralNetwork(
            data.layerSizes[0],
            data.layerSizes.slice(1, -1),
            data.layerSizes[data.layerSizes.length - 1],
            {
                learningRate: data.learningRate ?? 0.001,
                dropoutRate:  data.dropoutRate  ?? 0.0,
                useBatchNorm: data.useBatchNorm ?? false,
            }
        );
        net.W = data.W;
        net.b = data.b;
        net._syncLegacyWeights();
        return net;
    }
}
