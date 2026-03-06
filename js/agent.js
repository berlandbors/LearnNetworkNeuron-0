/**
 * agent.js — базовый класс агента, управляемого нейронной сетью.
 *
 * Агент движется по лабиринту, получает сенсорные входы
 * и накапливает оценку приспособленности (fitness).
 */
import { NeuralNetwork } from './neural-network.js';

/** Размер входного вектора (должен совпадать с getInputs()) */
export const INPUT_SIZE = 24;
/** Количество возможных действий: вверх, вниз, влево, вправо */
export const OUTPUT_SIZE = 4;

/** Направления движения: [dy, dx] */
const DIRECTIONS = [
    { dy: -1, dx:  0 }, // 0: вверх
    { dy:  1, dx:  0 }, // 1: вниз
    { dy:  0, dx: -1 }, // 2: влево
    { dy:  0, dx:  1 }, // 3: вправо
];

/** 8 направлений сенсоров (кардинальные + диагонали) */
const SENSOR_DIRS = [
    { dy: -1, dx:  0 },
    { dy:  1, dx:  0 },
    { dy:  0, dx: -1 },
    { dy:  0, dx:  1 },
    { dy: -1, dx: -1 },
    { dy: -1, dx:  1 },
    { dy:  1, dx: -1 },
    { dy:  1, dx:  1 },
];

export class Agent {
    /**
     * @param {NeuralNetwork|null} brain - готовый мозг или null (создаётся новый)
     * @param {number} hiddenSize - размер скрытого слоя
     */
    constructor(brain = null, hiddenSize = 16) {
        this.brain = brain || new NeuralNetwork(INPUT_SIZE, hiddenSize, OUTPUT_SIZE);

        /** @type {{x:number,y:number}} текущая позиция */
        this.pos = { x: 1, y: 1 };

        /** @type {number[][]} лабиринт */
        this.maze = null;

        /** @type {{x:number,y:number}} цель */
        this.goal = { x: 1, y: 1 };

        /** Размер лабиринта */
        this.mazeWidth = 1;
        this.mazeHeight = 1;

        /** Количество шагов */
        this.steps = 0;

        /** Накопленная оценка */
        this.fitness = 0;

        /** Минимальное расстояние до цели (для фитнеса) */
        this.minDistToGoal = Infinity;

        /** Агент достиг цели */
        this.reached = false;

        /** Путь агента (для визуализации) */
        this.path = [];

        /** Посещённые клетки */
        this.visited = new Set();

        /** Последние 4 действия (one-hot: 4×4=16 бит) */
        this.lastActions = [0, 0, 0, 0];

        /** Расстояние до цели в начале */
        this.startDist = 0;
    }

    /**
     * Установить параметры среды.
     * @param {number[][]} maze
     * @param {{x:number,y:number}} start
     * @param {{x:number,y:number}} goal
     */
    setEnvironment(maze, start, goal) {
        this.maze = maze;
        this.mazeHeight = maze.length;
        this.mazeWidth = maze[0].length;
        this.goal = goal;
        this.reset(start);
    }

    /**
     * Сбросить состояние агента в начальную позицию.
     * @param {{x:number,y:number}} [startPos]
     */
    reset(startPos) {
        if (startPos) this.pos = { ...startPos };
        this.steps = 0;
        this.fitness = 0;
        this.reached = false;
        this.path = [{ ...this.pos }];
        this.visited = new Set([`${this.pos.x},${this.pos.y}`]);
        this.lastActions = [0, 0, 0, 0];
        this.startDist = this._distToGoal();
        this.minDistToGoal = this.startDist;
    }

    /** Евклидово расстояние до цели */
    _distToGoal() {
        const dx = this.pos.x - this.goal.x;
        const dy = this.pos.y - this.goal.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Расстояние до ближайшей стены в заданном направлении.
     * @param {number} dy
     * @param {number} dx
     * @returns {number} нормализованное [0, 1]
     */
    _sensorDist(dy, dx) {
        let r = this.pos.y;
        let c = this.pos.x;
        let dist = 0;
        const maxDist = Math.max(this.mazeWidth, this.mazeHeight);

        while (true) {
            r += dy;
            c += dx;
            dist++;
            if (r < 0 || r >= this.mazeHeight || c < 0 || c >= this.mazeWidth ||
                this.maze[r][c] === 1) {
                break;
            }
            if (dist >= maxDist) break;
        }

        return dist / maxDist;
    }

    /**
     * Получить входной вектор для нейронной сети.
     * Размерность: INPUT_SIZE = 24
     *
     * [0..7]  - 8 расстояний до стен (нормализованных)
     * [8..9]  - нормализованная позиция агента (x/w, y/h)
     * [10..11] - нормализованная позиция цели (x/w, y/h)
     * [12..13] - угол к цели (cos, sin)
     * [14]    - нормализованное расстояние до цели
     * [15..18] - последние 4 действия (one-hot)
     * [19..22] - стена слева/справа/сверху/снизу (0/1)
     * [23]    - нормализованное количество шагов
     * @returns {number[]}
     */
    getInputs() {
        const inputs = [];

        // 1. Расстояния до стен по 8 направлениям
        for (const { dy, dx } of SENSOR_DIRS) {
            inputs.push(this._sensorDist(dy, dx));
        }

        // 2. Нормализованная позиция агента
        inputs.push(this.pos.x / (this.mazeWidth - 1));
        inputs.push(this.pos.y / (this.mazeHeight - 1));

        // 3. Нормализованная позиция цели
        inputs.push(this.goal.x / (this.mazeWidth - 1));
        inputs.push(this.goal.y / (this.mazeHeight - 1));

        // 4. Угол к цели
        const dx = this.goal.x - this.pos.x;
        const dy = this.goal.y - this.pos.y;
        const angle = Math.atan2(dy, dx);
        inputs.push(Math.cos(angle));
        inputs.push(Math.sin(angle));

        // 5. Нормализованное расстояние до цели
        const maxD = Math.sqrt(
            (this.mazeWidth - 1) ** 2 + (this.mazeHeight - 1) ** 2
        );
        inputs.push(this._distToGoal() / maxD);

        // 6. Последние 4 действия
        inputs.push(...this.lastActions);

        // 7. Стены по 4 основным направлениям
        for (const { dy: ddy, dx: ddx } of DIRECTIONS) {
            const ny = this.pos.y + ddy;
            const nx = this.pos.x + ddx;
            const isWall = (ny < 0 || ny >= this.mazeHeight ||
                            nx < 0 || nx >= this.mazeWidth ||
                            this.maze[ny][nx] === 1) ? 1 : 0;
            inputs.push(isWall);
        }

        // 8. Нормализованное количество шагов (будем нормировать на 500)
        inputs.push(Math.min(this.steps / 500, 1));

        return inputs;
    }

    /**
     * Выполнить одно действие с использованием нейросети.
     */
    think() {
        const inputs = this.getInputs();
        const outputs = this.brain.predict(inputs);
        // Выбрать действие с максимальным выходом
        const actionIndex = outputs.indexOf(Math.max(...outputs));
        this.move(actionIndex);
    }

    /**
     * Переместить агента.
     * @param {number} [actionIndex] - 0:вверх 1:вниз 2:влево 3:вправо
     *                                 Если не задан — агент думает сам
     */
    move(actionIndex) {
        if (this.reached) return;

        if (actionIndex === undefined) {
            const inputs = this.getInputs();
            const outputs = this.brain.predict(inputs);
            actionIndex = outputs.indexOf(Math.max(...outputs));
        }

        const prevDist = this._distToGoal();

        const { dy, dx } = DIRECTIONS[actionIndex];
        const ny = this.pos.y + dy;
        const nx = this.pos.x + dx;

        // Проверить, заблокировано ли движение стеной или границей
        const blocked = (
            ny < 0 || ny >= this.mazeHeight ||
            nx < 0 || nx >= this.mazeWidth ||
            this.maze[ny][nx] === 1
        );

        if (!blocked) {
            this.pos = { x: nx, y: ny };
            this.path.push({ ...this.pos });
            this.visited.add(`${nx},${ny}`);
        }

        // Обновить последние действия
        this.lastActions = [
            actionIndex === 0 ? 1 : 0,
            actionIndex === 1 ? 1 : 0,
            actionIndex === 2 ? 1 : 0,
            actionIndex === 3 ? 1 : 0,
        ];

        this.steps++;

        const newDist = this._distToGoal();
        if (newDist < this.minDistToGoal) this.minDistToGoal = newDist;

        // Проверить достижение цели
        if (this.pos.x === this.goal.x && this.pos.y === this.goal.y) {
            this.reached = true;
        }

        // Накопить награду
        this.fitness += this.calculateReward(prevDist, newDist, blocked);
    }

    /**
     * Вычислить награду за текущий шаг.
     * @param {number}  prevDist - расстояние до цели до шага
     * @param {number}  newDist  - расстояние до цели после шага
     * @param {boolean} blocked  - движение было заблокировано стеной
     * @returns {number}
     */
    calculateReward(prevDist, newDist, blocked) {
        if (this.reached) return 1000;

        let reward = 0;

        // Приближение / удаление от цели
        const distDelta = prevDist - newDist;
        if (distDelta > 0) {
            reward += 10 * distDelta;
        } else {
            reward += 5 * distDelta; // отрицательно
        }

        // Штраф за столкновение со стеной
        if (blocked) reward -= 10;

        // Штраф за каждый шаг
        reward -= 0.1;

        // Штраф за посещённую клетку
        const key = `${this.pos.x},${this.pos.y}`;
        const visits = [...this.visited].filter(k => k === key).length;
        if (visits > 1) reward -= 2;

        return reward;
    }

    /**
     * Финальная оценка агента после завершения эпизода.
     * @param {number} maxSteps
     * @returns {number}
     */
    calculateFitness(maxSteps) {
        let score = this.fitness;

        // Бонус за достижение цели
        if (this.reached) {
            score += 5000 + (maxSteps - this.steps) * 10;
        } else {
            // Бонус за приближение к цели
            if (this.startDist > 0) {
                score += (1 - this.minDistToGoal / this.startDist) * 500;
            }
        }

        return score;
    }
}
