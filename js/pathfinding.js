/**
 * pathfinding.js — алгоритмы поиска пути в лабиринте.
 * Реализованы A* и Dijkstra.
 */
export class Pathfinding {
    /**
     * Эвристика Манхэттенского расстояния.
     * @param {{x:number,y:number}} a
     * @param {{x:number,y:number}} b
     * @returns {number}
     */
    static heuristic(a, b) {
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    }

    /**
     * Получить проходимых соседей клетки в лабиринте.
     * @param {{x:number,y:number}} pos
     * @param {number[][]} maze
     * @returns {{x:number,y:number}[]}
     */
    static getNeighbors(pos, maze) {
        const dirs = [
            { x: 0, y: -1 },
            { x: 0, y:  1 },
            { x: -1, y: 0 },
            { x:  1, y: 0 },
        ];
        const neighbors = [];
        const h = maze.length;
        const w = maze[0].length;

        for (const d of dirs) {
            const nx = pos.x + d.x;
            const ny = pos.y + d.y;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h && maze[ny][nx] === 0) {
                neighbors.push({ x: nx, y: ny });
            }
        }

        return neighbors;
    }

    /**
     * Восстановить путь по карте предшественников.
     * @param {Map<string,{x:number,y:number}>} cameFrom
     * @param {{x:number,y:number}} current
     * @returns {{x:number,y:number}[]}
     */
    static reconstructPath(cameFrom, current) {
        const path = [current];
        let key = `${current.x},${current.y}`;

        while (cameFrom.has(key)) {
            current = cameFrom.get(key);
            key = `${current.x},${current.y}`;
            path.unshift(current);
        }

        return path;
    }

    /**
     * Алгоритм A* — поиск кратчайшего пути в лабиринте.
     * @param {number[][]} maze
     * @param {{x:number,y:number}} start
     * @param {{x:number,y:number}} goal
     * @returns {{x:number,y:number}[]|null} путь или null если нет пути
     */
    static aStar(maze, start, goal) {
        /** @type {Map<string, number>} */
        const gScore = new Map();
        /** @type {Map<string, number>} */
        const fScore = new Map();
        /** @type {Map<string, {x:number,y:number}>} */
        const cameFrom = new Map();
        /** @type {Set<string>} */
        const closedSet = new Set();

        const startKey = `${start.x},${start.y}`;
        gScore.set(startKey, 0);
        fScore.set(startKey, Pathfinding.heuristic(start, goal));

        // Простая реализация открытого списка на массиве (достаточно для наших размеров)
        const openList = [{ pos: start, f: fScore.get(startKey) }];

        while (openList.length > 0) {
            // Найти узел с наименьшим f
            openList.sort((a, b) => a.f - b.f);
            const { pos: current } = openList.shift();
            const currentKey = `${current.x},${current.y}`;

            if (current.x === goal.x && current.y === goal.y) {
                return Pathfinding.reconstructPath(cameFrom, current);
            }

            closedSet.add(currentKey);

            for (const neighbor of Pathfinding.getNeighbors(current, maze)) {
                const neighborKey = `${neighbor.x},${neighbor.y}`;
                if (closedSet.has(neighborKey)) continue;

                const tentativeG = (gScore.get(currentKey) ?? Infinity) + 1;

                if (tentativeG < (gScore.get(neighborKey) ?? Infinity)) {
                    cameFrom.set(neighborKey, current);
                    gScore.set(neighborKey, tentativeG);
                    const f = tentativeG + Pathfinding.heuristic(neighbor, goal);
                    fScore.set(neighborKey, f);

                    // Добавить в открытый список если ещё нет
                    if (!openList.some(n => n.pos.x === neighbor.x && n.pos.y === neighbor.y)) {
                        openList.push({ pos: neighbor, f });
                    }
                }
            }
        }

        return null; // Путь не найден
    }

    /**
     * Алгоритм A* с пошаговой визуализацией.
     * callback вызывается на каждом шаге с текущим состоянием.
     * @param {number[][]} maze
     * @param {{x:number,y:number}} start
     * @param {{x:number,y:number}} goal
     * @param {function({visited:Set<string>, path:{x:number,y:number}[]}):void} callback
     * @returns {{x:number,y:number}[]|null}
     */
    static aStarVisualized(maze, start, goal, callback) {
        const gScore = new Map();
        const fScore = new Map();
        const cameFrom = new Map();
        const closedSet = new Set();

        const startKey = `${start.x},${start.y}`;
        gScore.set(startKey, 0);
        fScore.set(startKey, Pathfinding.heuristic(start, goal));

        const openList = [{ pos: start, f: fScore.get(startKey) }];

        while (openList.length > 0) {
            openList.sort((a, b) => a.f - b.f);
            const { pos: current } = openList.shift();
            const currentKey = `${current.x},${current.y}`;

            callback({ visited: new Set(closedSet), current });

            if (current.x === goal.x && current.y === goal.y) {
                const path = Pathfinding.reconstructPath(cameFrom, current);
                callback({ visited: new Set(closedSet), current, path });
                return path;
            }

            closedSet.add(currentKey);

            for (const neighbor of Pathfinding.getNeighbors(current, maze)) {
                const neighborKey = `${neighbor.x},${neighbor.y}`;
                if (closedSet.has(neighborKey)) continue;

                const tentativeG = (gScore.get(currentKey) ?? Infinity) + 1;

                if (tentativeG < (gScore.get(neighborKey) ?? Infinity)) {
                    cameFrom.set(neighborKey, current);
                    gScore.set(neighborKey, tentativeG);
                    const f = tentativeG + Pathfinding.heuristic(neighbor, goal);
                    fScore.set(neighborKey, f);

                    if (!openList.some(n => n.pos.x === neighbor.x && n.pos.y === neighbor.y)) {
                        openList.push({ pos: neighbor, f });
                    }
                }
            }
        }

        return null;
    }

    /**
     * Алгоритм Дейкстры — поиск кратчайшего пути (без эвристики).
     * @param {number[][]} maze
     * @param {{x:number,y:number}} start
     * @param {{x:number,y:number}} goal
     * @returns {{x:number,y:number}[]|null}
     */
    static dijkstra(maze, start, goal) {
        const dist = new Map();
        const cameFrom = new Map();
        const visited = new Set();

        const startKey = `${start.x},${start.y}`;
        dist.set(startKey, 0);

        const queue = [{ pos: start, d: 0 }];

        while (queue.length > 0) {
            queue.sort((a, b) => a.d - b.d);
            const { pos: current } = queue.shift();
            const currentKey = `${current.x},${current.y}`;

            if (visited.has(currentKey)) continue;
            visited.add(currentKey);

            if (current.x === goal.x && current.y === goal.y) {
                return Pathfinding.reconstructPath(cameFrom, current);
            }

            for (const neighbor of Pathfinding.getNeighbors(current, maze)) {
                const neighborKey = `${neighbor.x},${neighbor.y}`;
                if (visited.has(neighborKey)) continue;

                const newDist = (dist.get(currentKey) ?? Infinity) + 1;
                if (newDist < (dist.get(neighborKey) ?? Infinity)) {
                    dist.set(neighborKey, newDist);
                    cameFrom.set(neighborKey, current);
                    queue.push({ pos: neighbor, d: newDist });
                }
            }
        }

        return null;
    }
}
