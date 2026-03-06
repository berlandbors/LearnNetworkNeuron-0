/**
 * NeuralNetwork - простая нейронная сеть прямого распространения.
 * Поддерживает генетические операции (кроссовер, мутация) и сериализацию.
 */
export class NeuralNetwork {
    /**
     * @param {number} inputSize  - количество входных нейронов
     * @param {number} hiddenSize - количество нейронов в скрытом слое
     * @param {number} outputSize - количество выходных нейронов
     */
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Инициализируем веса случайными значениями
        this.weightsIH = this.randomMatrix(hiddenSize, inputSize);
        this.weightsHO = this.randomMatrix(outputSize, hiddenSize);
        this.biasH = this.randomArray(hiddenSize);
        this.biasO = this.randomArray(outputSize);
    }

    /**
     * Создать матрицу rows×cols со случайными значениями в [-1, 1].
     * @param {number} rows
     * @param {number} cols
     * @returns {number[][]}
     */
    randomMatrix(rows, cols) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    /**
     * Создать массив размером size со случайными значениями в [-1, 1].
     * @param {number} size
     * @returns {number[]}
     */
    randomArray(size) {
        return Array.from({ length: size }, () => Math.random() * 2 - 1);
    }

    /**
     * Сигмоид-активация.
     * @param {number} x
     * @returns {number}
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * ReLU-активация.
     * @param {number} x
     * @returns {number}
     */
    relu(x) {
        return Math.max(0, x);
    }

    /**
     * Прямой проход нейронной сети.
     * Скрытый слой — ReLU, выходной — Sigmoid.
     * @param {number[]} inputs - входные данные
     * @returns {number[]} - массив выходных значений [0, 1]
     */
    predict(inputs) {
        // Скрытый слой
        const hidden = this.biasH.map((b, i) => {
            const sum = this.weightsIH[i].reduce((acc, w, j) => acc + w * (inputs[j] ?? 0), b);
            return this.relu(sum);
        });

        // Выходной слой
        const output = this.biasO.map((b, i) => {
            const sum = this.weightsHO[i].reduce((acc, w, j) => acc + w * hidden[j], b);
            return this.sigmoid(sum);
        });

        return output;
    }

    /**
     * Создать глубокую копию этой сети.
     * @returns {NeuralNetwork}
     */
    copy() {
        const nn = new NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize);
        nn.weightsIH = this.weightsIH.map(row => [...row]);
        nn.weightsHO = this.weightsHO.map(row => [...row]);
        nn.biasH = [...this.biasH];
        nn.biasO = [...this.biasO];
        return nn;
    }

    /**
     * Мутировать веса сети с заданной вероятностью.
     * @param {number} rate - вероятность мутации каждого веса (0–1)
     */
    mutate(rate) {
        const mutateVal = v =>
            Math.random() < rate ? v + (Math.random() * 2 - 1) * 0.5 : v;

        this.weightsIH = this.weightsIH.map(row => row.map(mutateVal));
        this.weightsHO = this.weightsHO.map(row => row.map(mutateVal));
        this.biasH = this.biasH.map(mutateVal);
        this.biasO = this.biasO.map(mutateVal);
    }

    /**
     * Кроссовер (равномерный): смешать веса с партнёром.
     * @param {NeuralNetwork} partner
     * @returns {NeuralNetwork} - дочерняя нейронная сеть
     */
    crossover(partner) {
        const child = new NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize);

        child.weightsIH = this.weightsIH.map((row, i) =>
            row.map((val, j) => Math.random() < 0.5 ? val : partner.weightsIH[i][j])
        );
        child.weightsHO = this.weightsHO.map((row, i) =>
            row.map((val, j) => Math.random() < 0.5 ? val : partner.weightsHO[i][j])
        );
        child.biasH = this.biasH.map((val, i) =>
            Math.random() < 0.5 ? val : partner.biasH[i]
        );
        child.biasO = this.biasO.map((val, i) =>
            Math.random() < 0.5 ? val : partner.biasO[i]
        );

        return child;
    }

    /**
     * Сериализовать сеть в JSON-объект.
     * @returns {object}
     */
    serialize() {
        return {
            inputSize: this.inputSize,
            hiddenSize: this.hiddenSize,
            outputSize: this.outputSize,
            weightsIH: this.weightsIH,
            weightsHO: this.weightsHO,
            biasH: this.biasH,
            biasO: this.biasO,
        };
    }

    /**
     * Восстановить сеть из сериализованных данных.
     * @param {object} data
     * @returns {NeuralNetwork}
     */
    static deserialize(data) {
        const nn = new NeuralNetwork(data.inputSize, data.hiddenSize, data.outputSize);
        nn.weightsIH = data.weightsIH;
        nn.weightsHO = data.weightsHO;
        nn.biasH = data.biasH;
        nn.biasO = data.biasO;
        return nn;
    }
}
