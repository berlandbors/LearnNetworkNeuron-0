/**
 * renderer.js — отрисовка лабиринта, агентов, путей и сенсоров на Canvas.
 */

/** Цвета */
const COLORS = {
    wall:        '#1a1a2e',
    floor:       '#16213e',
    grid:        'rgba(255,255,255,0.04)',
    goal:        '#f59e0b',
    optimalPath: 'rgba(251, 191, 36, 0.25)',
    agentPath:   'rgba(102, 126, 234, 0.3)',
    agent:       '#667eea',
    sensor:      'rgba(102, 234, 160, 0.4)',
    text:        'rgba(255,255,255,0.9)',
};

export class Renderer {
    /**
     * @param {HTMLCanvasElement} canvas - главный холст
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.cellSize = 30;
        this.maze = null;
    }

    /** Задать размер клетки */
    setCellSize(size) {
        this.cellSize = size;
    }

    /** Полная очистка холста */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Нарисовать лабиринт.
     * @param {number[][]} maze
     */
    drawMaze(maze) {
        this.maze = maze;
        const { ctx, cellSize: cs } = this;
        const rows = maze.length;
        const cols = maze[0].length;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                ctx.fillStyle = maze[r][c] === 1 ? COLORS.wall : COLORS.floor;
                ctx.fillRect(c * cs, r * cs, cs, cs);
            }
        }
    }

    /**
     * Нарисовать тонкую сетку поверх лабиринта.
     */
    drawGrid() {
        if (!this.maze) return;
        const { ctx, cellSize: cs } = this;
        const rows = this.maze.length;
        const cols = this.maze[0].length;

        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.5;
        for (let r = 0; r <= rows; r++) {
            ctx.beginPath();
            ctx.moveTo(0, r * cs);
            ctx.lineTo(cols * cs, r * cs);
            ctx.stroke();
        }
        for (let c = 0; c <= cols; c++) {
            ctx.beginPath();
            ctx.moveTo(c * cs, 0);
            ctx.lineTo(c * cs, rows * cs);
            ctx.stroke();
        }
    }

    /**
     * Нарисовать цель (звезда / флажок).
     * @param {{x:number,y:number}} goal
     */
    drawGoal(goal) {
        const { ctx, cellSize: cs } = this;
        const cx = goal.x * cs + cs / 2;
        const cy = goal.y * cs + cs / 2;
        const r = cs * 0.35;

        // Пульсирующий круг
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.goal;
        ctx.shadowColor = COLORS.goal;
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.restore();

        // Буква G
        ctx.fillStyle = '#1a1a2e';
        ctx.font = `bold ${cs * 0.45}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('G', cx, cy);
    }

    /**
     * Нарисовать агента.
     * @param {import('./agent.js').Agent} agent
     * @param {string} [color]
     */
    drawAgent(agent, color = COLORS.agent) {
        const { ctx, cellSize: cs } = this;
        const cx = agent.pos.x * cs + cs / 2;
        const cy = agent.pos.y * cs + cs / 2;
        const r = cs * 0.35;

        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.restore();

        // Буква A
        ctx.fillStyle = '#fff';
        ctx.font = `bold ${cs * 0.38}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('A', cx, cy);
    }

    /**
     * Нарисовать путь агента с затуханием.
     * @param {{x:number,y:number}[]} path
     * @param {string} [color]
     */
    drawPath(path, color = COLORS.agentPath) {
        if (!path || path.length < 2) return;
        const { ctx, cellSize: cs } = this;
        const half = cs / 2;

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = cs * 0.15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(path[0].x * cs + half, path[0].y * cs + half);
        for (let i = 1; i < path.length; i++) {
            ctx.lineTo(path[i].x * cs + half, path[i].y * cs + half);
        }
        ctx.stroke();
        ctx.restore();
    }

    /**
     * Нарисовать оптимальный путь (A*) пунктиром.
     * @param {{x:number,y:number}[]|null} path
     */
    drawOptimalPath(path) {
        if (!path || path.length < 2) return;
        const { ctx, cellSize: cs } = this;
        const half = cs / 2;

        ctx.save();
        ctx.strokeStyle = COLORS.optimalPath;
        ctx.lineWidth = cs * 0.2;
        ctx.lineCap = 'round';
        ctx.setLineDash([cs * 0.3, cs * 0.15]);
        ctx.beginPath();
        ctx.moveTo(path[0].x * cs + half, path[0].y * cs + half);
        for (let i = 1; i < path.length; i++) {
            ctx.lineTo(path[i].x * cs + half, path[i].y * cs + half);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
    }

    /**
     * Показать текст об эффективности агента.
     * @param {number} aiSteps      - шаги агента
     * @param {number} optimalSteps - шаги A*
     */
    drawEfficiency(aiSteps, optimalSteps) {
        if (!optimalSteps) return;
        const { ctx, canvas } = this;
        const ratio = (optimalSteps > 0 && aiSteps > 0) ? (optimalSteps / aiSteps * 100).toFixed(1) : '—';
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(canvas.width - 160, 5, 155, 50);
        ctx.fillStyle = COLORS.text;
        ctx.font = '12px monospace';
        ctx.fillText(`AI шагов: ${aiSteps}`, canvas.width - 150, 22);
        ctx.fillText(`Оптим.: ${optimalSteps}`, canvas.width - 150, 38);
        ctx.fillText(`Эффект.: ${ratio}%`, canvas.width - 150, 54);
        ctx.restore();
    }

    /**
     * Визуализировать сенсоры агента (лучи до стен).
     * @param {import('./agent.js').Agent} agent
     */
    drawSensors(agent) {
        if (!agent.maze) return;
        const { ctx, cellSize: cs } = this;
        const cx = agent.pos.x * cs + cs / 2;
        const cy = agent.pos.y * cs + cs / 2;

        const sensorDirs = [
            { dy: -1, dx:  0 }, { dy:  1, dx:  0 },
            { dy:  0, dx: -1 }, { dy:  0, dx:  1 },
            { dy: -1, dx: -1 }, { dy: -1, dx:  1 },
            { dy:  1, dx: -1 }, { dy:  1, dx:  1 },
        ];

        ctx.save();
        ctx.strokeStyle = COLORS.sensor;
        ctx.lineWidth = 0.8;

        for (const { dy, dx } of sensorDirs) {
            let r = agent.pos.y;
            let c = agent.pos.x;
            const maxDist = Math.max(agent.mazeWidth, agent.mazeHeight);
            let dist = 0;

            while (dist < maxDist) {
                r += dy;
                c += dx;
                dist++;
                if (r < 0 || r >= agent.mazeHeight ||
                    c < 0 || c >= agent.mazeWidth ||
                    agent.maze[r][c] === 1) break;
            }

            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(c * cs + cs / 2, r * cs + cs / 2);
            ctx.stroke();
        }

        ctx.restore();
    }

    /**
     * (Опционально) Визуализировать нейронную сеть мини-схемой.
     * @param {import('./neural-network.js').NeuralNetwork} brain
     * @param {number} x - верхний левый угол схемы
     * @param {number} y
     */
    drawNeuralNetwork(brain, x, y) {
        const { ctx } = this;
        const layers = [brain.inputSize, brain.hiddenSize, brain.outputSize];
        const layerGap = 60;
        const maxN = Math.max(...layers);
        const nodeR = 5;
        const totalH = maxN * 14;

        ctx.save();
        ctx.globalAlpha = 0.7;

        // Рисовать соединения (только часть для читаемости)
        const positions = layers.map((n, li) => {
            const colX = x + li * layerGap;
            return Array.from({ length: n }, (_, ni) => ({
                x: colX,
                y: y + (ni + 0.5) * (totalH / n),
            }));
        });

        ctx.strokeStyle = 'rgba(100,100,255,0.15)';
        ctx.lineWidth = 0.5;
        for (let li = 0; li < layers.length - 1; li++) {
            const from = positions[li].slice(0, Math.min(8, layers[li]));
            const to = positions[li + 1].slice(0, Math.min(8, layers[li + 1]));
            for (const f of from) {
                for (const t of to) {
                    ctx.beginPath();
                    ctx.moveTo(f.x, f.y);
                    ctx.lineTo(t.x, t.y);
                    ctx.stroke();
                }
            }
        }

        // Рисовать узлы
        for (let li = 0; li < layers.length; li++) {
            const nodeColor = li === 0 ? '#60a5fa' : li === 1 ? '#a78bfa' : '#4ade80';
            const visible = Math.min(8, layers[li]);
            for (let ni = 0; ni < visible; ni++) {
                const { x: nx, y: ny } = positions[li][ni];
                ctx.beginPath();
                ctx.arc(nx, ny, nodeR, 0, Math.PI * 2);
                ctx.fillStyle = nodeColor;
                ctx.fill();
            }
            if (layers[li] > 8) {
                ctx.fillStyle = 'rgba(255,255,255,0.5)';
                ctx.font = '9px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`+${layers[li] - 8}`, x + li * layerGap, y + totalH + 12);
            }
        }

        ctx.restore();
    }

    /**
     * Подогнать размер canvas под лабиринт.
     * @param {number[][]} maze
     * @param {number}     cellSize
     */
    fitCanvas(maze, cellSize) {
        this.cellSize = cellSize;
        this.canvas.width  = maze[0].length * cellSize;
        this.canvas.height = maze.length    * cellSize;
    }

    /**
     * Нарисовать тепловую карту глобальной памяти (интенсивность посещений).
     * @param {Map<string,number>} globalVisitCount
     */
    drawGlobalMemoryHeatmap(globalVisitCount) {
        if (!globalVisitCount || globalVisitCount.size === 0) return;

        const { ctx, cellSize: cs, maze } = this;
        const maxVisits = Math.max(...globalVisitCount.values());

        for (const [key, count] of globalVisitCount.entries()) {
            const [x, y] = key.split(',').map(Number);

            if (maze[y] && maze[y][x] === 0) {
                const intensity = count / maxVisits;
                ctx.fillStyle = `rgba(255, 150, 0, ${intensity * 0.3})`;
                ctx.fillRect(x * cs, y * cs, cs, cs);
            }
        }
    }
}
