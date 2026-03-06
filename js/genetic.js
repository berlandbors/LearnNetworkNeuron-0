/**
 * genetic.js — генетический алгоритм для эволюции нейронных сетей.
 *
 * Методы: турнирная селекция, элитизм, кроссовер, мутация.
 */
import { Agent, INPUT_SIZE, OUTPUT_SIZE } from './agent.js';
import { NeuralNetwork } from './neural-network.js';

export class GeneticAlgorithm {
    /**
     * @param {number} populationSize - размер популяции
     * @param {number} mutationRate   - вероятность мутации (0–1)
     * @param {number} hiddenSize     - нейронов в скрытом слое
     */
    constructor(populationSize = 50, mutationRate = 0.1, hiddenSize = 16) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.hiddenSize = hiddenSize;
    }

    /**
     * Инициализировать популяцию случайными агентами.
     * @returns {Agent[]}
     */
    initPopulation() {
        return Array.from({ length: this.populationSize }, () =>
            new Agent(null, this.hiddenSize)
        );
    }

    /**
     * Турнирная селекция: выбрать лучшего из k случайных агентов.
     * @param {Agent[]} population
     * @param {number} [k=5] - размер турнира
     * @returns {Agent}
     */
    selectParent(population, k = 5) {
        let best = null;
        for (let i = 0; i < k; i++) {
            const candidate = population[Math.floor(Math.random() * population.length)];
            if (best === null || candidate.fitness > best.fitness) {
                best = candidate;
            }
        }
        return best;
    }

    /**
     * Кроссовер двух агентов → дочерний агент.
     * @param {Agent} parent1
     * @param {Agent} parent2
     * @returns {Agent}
     */
    crossover(parent1, parent2) {
        const childBrain = parent1.brain.crossover(parent2.brain);
        return new Agent(childBrain, this.hiddenSize);
    }

    /**
     * Мутировать агента (in-place).
     * @param {Agent} agent
     * @param {number} [rate] - переопределить mutationRate
     */
    mutate(agent, rate) {
        agent.brain.mutate(rate ?? this.mutationRate);
    }

    /**
     * Создать следующее поколение.
     * Шаги:
     *  1. Вычислить фитнес каждого агента.
     *  2. Отсортировать по убыванию фитнеса.
     *  3. Элитизм — 2 лучших переходят без изменений.
     *  4. Остальные — кроссовер + мутация.
     *
     * @param {Agent[]} population - текущая популяция (уже завершила эпизод)
     * @param {number}  maxSteps   - для вычисления финального фитнеса
     * @returns {Agent[]} - новая популяция
     */
    nextGeneration(population, maxSteps = 500) {
        // 1. Вычислить и записать финальный фитнес
        for (const agent of population) {
            agent.fitness = agent.calculateFitness(maxSteps);
        }

        // 2. Сортировка
        population.sort((a, b) => b.fitness - a.fitness);

        const newPopulation = [];

        // 3. Элитизм — 2 лучших
        const eliteCount = Math.min(2, population.length);
        for (let i = 0; i < eliteCount; i++) {
            const elite = new Agent(population[i].brain.copy(), this.hiddenSize);
            newPopulation.push(elite);
        }

        // 4. Кроссовер + мутация для остальных
        while (newPopulation.length < this.populationSize) {
            const parent1 = this.selectParent(population);
            const parent2 = this.selectParent(population);
            const child = this.crossover(parent1, parent2);
            this.mutate(child);
            newPopulation.push(child);
        }

        return newPopulation;
    }
}
