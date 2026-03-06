/**
 * maze.js — генерация лабиринтов и предустановленные уровни.
 *
 * Лабиринт представлен двумерным массивом:
 *   0 — свободная клетка
 *   1 — стена
 */

// ─── Предустановленные лабиринты ────────────────────────────────────────────

/** Простой лабиринт 10×10 */
const EASY_MAZE = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,0,1],
    [1,0,1,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1,1],
];

/** Средний лабиринт 15×15 */
const MEDIUM_MAZE = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,0,1,1,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,0,1,1,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,0,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
    [1,1,1,0,1,1,1,0,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
];



// ─── Класс генератора лабиринтов ─────────────────────────────────────────────

export class MazeGenerator {
    /**
     * Простой лабиринт (10×10).
     * @returns {number[][]}
     */
    static easy() {
        return EASY_MAZE.map(row => [...row]);
    }

    /**
     * Средний лабиринт (15×15).
     * @returns {number[][]}
     */
    static medium() {
        return MEDIUM_MAZE.map(row => [...row]);
    }

    /**
     * Сложный лабиринт (20×20), генерируется каждый раз.
     * @returns {number[][]}
     */
    static hard() {
        return MazeGenerator.randomBacktracker(20, 20);
    }

    /**
     * Случайный лабиринт с заданной плотностью стен.
     * Использует алгоритм Recursive Backtracker (DFS).
     * @param {number} width
     * @param {number} height
     * @param {number} [wallDensity] - не используется (для совместимости)
     * @returns {number[][]}
     */
    static random(width, height, wallDensity) {
        return MazeGenerator.randomBacktracker(width, height);
    }

    /**
     * Алгоритм Recursive Backtracker для генерации лабиринта.
     * Гарантирует наличие пути от любой клетки до любой другой.
     * @param {number} width  - нечётное → ширина с учётом стен
     * @param {number} height - нечётное → высота с учётом стен
     * @returns {number[][]}
     */
    static randomBacktracker(width, height) {
        // Работаем в «клеточных» координатах (каждая клетка разделена стенами)
        const cols = Math.floor(width / 2);
        const rows = Math.floor(height / 2);
        const w = cols * 2 + 1;
        const h = rows * 2 + 1;

        // Заполнить всё стенами
        const maze = Array.from({ length: h }, () => new Array(w).fill(1));

        // Отметить посещённые клетки
        const visited = Array.from({ length: rows }, () => new Array(cols).fill(false));

        const stack = [];
        const startR = 0;
        const startC = 0;
        visited[startR][startC] = true;
        stack.push([startR, startC]);

        // Открыть стартовую клетку
        maze[startR * 2 + 1][startC * 2 + 1] = 0;

        const dirs = [
            [-1, 0], [1, 0], [0, -1], [0, 1],
        ];

        while (stack.length > 0) {
            const [cr, cc] = stack[stack.length - 1];

            // Найти непосещённых соседей
            const neighbors = [];
            for (const [dr, dc] of dirs) {
                const nr = cr + dr;
                const nc = cc + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited[nr][nc]) {
                    neighbors.push([nr, nc, dr, dc]);
                }
            }

            if (neighbors.length === 0) {
                stack.pop();
            } else {
                // Случайный сосед
                const [nr, nc, dr, dc] = neighbors[Math.floor(Math.random() * neighbors.length)];
                visited[nr][nc] = true;

                // Открыть стену между текущей и следующей клеткой
                maze[cr * 2 + 1 + dr][cc * 2 + 1 + dc] = 0;
                // Открыть следующую клетку
                maze[nr * 2 + 1][nc * 2 + 1] = 0;

                stack.push([nr, nc]);
            }
        }

        return maze;
    }

    /**
     * Проверить, есть ли путь от start до goal в лабиринте (BFS).
     * @param {number[][]} maze
     * @param {{x:number,y:number}} start
     * @param {{x:number,y:number}} goal
     * @returns {boolean}
     */
    static isValidMaze(maze, start, goal) {
        const h = maze.length;
        const w = maze[0].length;
        const visited = Array.from({ length: h }, () => new Array(w).fill(false));
        const queue = [[start.y, start.x]];
        visited[start.y][start.x] = true;

        while (queue.length > 0) {
            const [r, c] = queue.shift();
            if (r === goal.y && c === goal.x) return true;

            for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                const nr = r + dr;
                const nc = c + dc;
                if (nr >= 0 && nr < h && nc >= 0 && nc < w &&
                    !visited[nr][nc] && maze[nr][nc] === 0) {
                    visited[nr][nc] = true;
                    queue.push([nr, nc]);
                }
            }
        }

        return false;
    }
}

// ─── Предустановленные лабиринты ────────────────────────────────────────────

export const PRESETS = {
    /** 10×10 простой */
    EASY: EASY_MAZE.map(row => [...row]),

    /** 15×15 средний */
    MEDIUM: MEDIUM_MAZE.map(row => [...row]),

    /** 20×20 сложный (генерируется один раз при загрузке модуля) */
    HARD: (() => {
        // Inline backtracker для избежания циклической зависимости
        const width = 20, height = 20;
        const cols = Math.floor(width / 2);
        const rows = Math.floor(height / 2);
        const w = cols * 2 + 1;
        const h = rows * 2 + 1;
        const maze = Array.from({ length: h }, () => new Array(w).fill(1));
        const visited = Array.from({ length: rows }, () => new Array(cols).fill(false));
        const stack = [[0, 0]];
        visited[0][0] = true;
        maze[1][1] = 0;
        const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
        while (stack.length > 0) {
            const [cr, cc] = stack[stack.length - 1];
            const nbrs = [];
            for (const [dr, dc] of dirs) {
                const nr = cr + dr, nc = cc + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited[nr][nc])
                    nbrs.push([nr, nc, dr, dc]);
            }
            if (nbrs.length === 0) { stack.pop(); }
            else {
                const [nr, nc, dr, dc] = nbrs[Math.floor(Math.random() * nbrs.length)];
                visited[nr][nc] = true;
                maze[cr*2+1+dr][cc*2+1+dc] = 0;
                maze[nr*2+1][nc*2+1] = 0;
                stack.push([nr, nc]);
            }
        }
        return maze;
    })(),
};
