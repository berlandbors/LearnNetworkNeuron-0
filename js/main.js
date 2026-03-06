/**
 * main.js — главный модуль приложения.
 *
 * Отвечает за:
 *  - инициализацию среды и популяции
 *  - главный цикл обучения (requestAnimationFrame)
 *  - обработку событий UI
 */

import { NeuralNetwork } from './neural-network.js';
import { Agent } from './agent.js';
import { RLAgent } from './rl-agent.js';
import { GeneticAlgorithm } from './genetic.js';
import { MazeGenerator, PRESETS } from './maze.js';
import { Renderer } from './renderer.js';
import { TrainingChart } from './chart.js';
import { Pathfinding } from './pathfinding.js';
import { Storage } from './storage.js';

// ─── Настройки по умолчанию ──────────────────────────────────────────────────

const config = {
    populationSize: 50,
    mutationRate: 0.1,
    hiddenSize: 16,
    maxSteps: 400,
    cellSize: 28,
    showSensors: false,
    showNeuralViz: false,
};

// ─── Глобальное состояние ────────────────────────────────────────────────────

const state = {
    mode: 'genetic',       // 'genetic' | 'rl'
    isTraining: false,
    speed: 1,              // 1 | 5 | 20
    generation: 0,
    population: [],
    currentAgentIndex: 0,
    bestFitness: -Infinity,
    bestAgent: null,
    maze: null,
    start: { x: 1, y: 1 },
    goal: { x: 1, y: 1 },
    optimalPath: null,
    rlAgent: null,
    animFrameId: null,
    showOptimal: true,
};

// ─── Canvas и рендерер ───────────────────────────────────────────────────────

const mazeCanvas  = document.getElementById('mazeCanvas');
const chartCanvas = document.getElementById('chartCanvas');
const renderer    = new Renderer(mazeCanvas);
const chart       = new TrainingChart(chartCanvas);

// ─── Инициализация ───────────────────────────────────────────────────────────

function init() {
    // Лабиринт
    const difficulty = document.getElementById('mazeDifficulty').value;
    state.maze = getMaze(difficulty);

    // Старт / Цель
    const rows = state.maze.length;
    const cols = state.maze[0].length;
    state.start = { x: 1, y: 1 };
    state.goal  = { x: cols - 2, y: rows - 2 };

    // Клетки старта/цели должны быть проходимы
    state.maze[state.start.y][state.start.x] = 0;
    state.maze[state.goal.y ][state.goal.x ] = 0;

    // Оптимальный путь (A*)
    state.optimalPath = Pathfinding.aStar(state.maze, state.start, state.goal);

    // Подогнать canvas
    renderer.fitCanvas(state.maze, config.cellSize);

    // Инициализация агентов
    readConfig();
    if (state.mode === 'genetic') {
        initGenetic();
    } else {
        initRL();
    }

    // Первичная отрисовка
    renderFrame();
    updateUI();
}

function getMaze(difficulty) {
    switch (difficulty) {
        case 'easy':   return MazeGenerator.easy();
        case 'medium': return MazeGenerator.medium();
        case 'hard':   return MazeGenerator.hard();
        default:       return MazeGenerator.random(20, 20);
    }
}

function initGenetic() {
    const ga = new GeneticAlgorithm(config.populationSize, config.mutationRate, config.hiddenSize);
    state.population = ga.initPopulation();
    state.population.forEach(a => a.setEnvironment(state.maze, state.start, state.goal));
    state.currentAgentIndex = 0;
    state.generation = 0;
    state.bestFitness = -Infinity;
    state.bestAgent = null;
}

function initRL() {
    const hiddenLayersInput = document.getElementById('hiddenLayers')?.value || '64,32';
    let hiddenLayers = hiddenLayersInput.split(',').map(x => parseInt(x.trim(), 10)).filter(n => Number.isFinite(n) && n > 0);
    if (hiddenLayers.length === 0) hiddenLayers = [64, 32]; // fallback default
    const useDueling           = document.getElementById('useDueling')?.checked ?? false;
    const useDoubleDQN         = document.getElementById('useDoubleDQN')?.checked ?? true;
    const usePrioritizedReplay = document.getElementById('usePrioritizedReplay')?.checked ?? true;
    const learningRate         = parseFloat(document.getElementById('learningRate')?.value ?? '0.001');

    state.rlAgent = new RLAgent(hiddenLayers, {
        dueling:           useDueling,
        doubleDQN:         useDoubleDQN,
        prioritizedReplay: usePrioritizedReplay,
        learningRate:      learningRate,
    });
    state.rlAgent.setEnvironment(state.maze, state.start, state.goal);
    state.generation = 0;
    state.bestFitness = -Infinity;
    state.bestAgent = null;
}

// ─── Главный цикл ────────────────────────────────────────────────────────────

function update() {
    if (!state.isTraining) return;

    const stepsPerFrame = getStepsPerFrame();

    if (state.mode === 'genetic') {
        for (let s = 0; s < stepsPerFrame; s++) {
            updateGenetic();
        }
    } else {
        for (let s = 0; s < stepsPerFrame; s++) {
            updateRL();
        }
    }

    renderFrame();
    updateUI();

    state.animFrameId = requestAnimationFrame(update);
}

function getStepsPerFrame() {
    switch (state.speed) {
        case 5:  return 5;
        case 20: return 20;
        default: return 1;
    }
}

// ─── Генетический режим ──────────────────────────────────────────────────────

function updateGenetic() {
    if (state.currentAgentIndex >= state.population.length) {
        nextGeneration();
        return;
    }

    const agent = state.population[state.currentAgentIndex];

    if (agent.reached || agent.steps >= config.maxSteps) {
        // Обновить лучший фитнес (промежуточный)
        const f = agent.calculateFitness(config.maxSteps);
        agent.fitness = f;
        if (f > state.bestFitness) {
            state.bestFitness = f;
            state.bestAgent = agent;
        }
        state.currentAgentIndex++;
        return;
    }

    agent.move();
}

function nextGeneration() {
    chart.update(state.generation, state.population);

    const ga = new GeneticAlgorithm(config.populationSize, config.mutationRate, config.hiddenSize);
    state.population = ga.nextGeneration(state.population, config.maxSteps);
    state.population.forEach(a => a.setEnvironment(state.maze, state.start, state.goal));
    state.currentAgentIndex = 0;
    state.generation++;

    // Лучший агент текущего поколения
    const best = state.population[0];
    if (best.fitness > state.bestFitness) {
        state.bestFitness = best.fitness;
        state.bestAgent = best;
    }
}

// ─── RL режим ────────────────────────────────────────────────────────────────

function updateRL() {
    const agent = state.rlAgent;
    const state_t = agent.getInputs();
    const action  = agent.act(state_t);

    const fitnessBefore = agent.fitness;
    agent.move(action);
    const stepReward = agent.fitness - fitnessBefore;

    const state_t1 = agent.getInputs();
    const done     = agent.reached || agent.steps >= config.maxSteps;

    agent.remember(state_t, action, stepReward, state_t1, done);

    if (done) {
        agent.replay(32);
        agent.updateEpsilon();

        if (agent.fitness > state.bestFitness) {
            state.bestFitness = agent.fitness;
            state.bestAgent = agent;
        }

        // Обновить график (pseudo-population для chart)
        chart.update(state.generation, [agent]);

        agent.reset(state.start);
        state.generation++;
    }
}

// ─── Отрисовка ───────────────────────────────────────────────────────────────

function renderFrame() {
    renderer.clear();
    renderer.drawMaze(state.maze);
    renderer.drawGrid();

    if (state.showOptimal && state.optimalPath) {
        renderer.drawOptimalPath(state.optimalPath);
    }

    renderer.drawGoal(state.goal);

    // Нарисовать текущего агента
    const agent = getCurrentAgent();
    if (agent) {
        renderer.drawPath(agent.path);
        if (config.showSensors) renderer.drawSensors(agent);
        renderer.drawAgent(agent);

        if (state.optimalPath) {
            renderer.drawEfficiency(agent.steps, state.optimalPath.length - 1);
        }
    }
}

function getCurrentAgent() {
    if (state.mode === 'genetic') {
        return state.population[Math.min(state.currentAgentIndex, state.population.length - 1)] || null;
    }
    return state.rlAgent;
}

// ─── Обновление UI ───────────────────────────────────────────────────────────

function updateUI() {
    document.getElementById('generationVal').textContent = state.generation;
    document.getElementById('bestFitnessVal').textContent =
        state.bestFitness === -Infinity ? '—' : state.bestFitness.toFixed(1);

    if (state.mode === 'genetic') {
        const agentIdx = Math.min(state.currentAgentIndex, state.population.length - 1);
        document.getElementById('currentAgentVal').textContent =
            `${agentIdx + 1} / ${state.population.length}`;

        const agent = state.population[agentIdx];
        if (agent) {
            document.getElementById('stepsVal').textContent = agent.steps;
            document.getElementById('reachedVal').textContent = agent.reached ? '✅' : '❌';
        }
    } else {
        const agent = state.rlAgent;
        if (agent) {
            document.getElementById('currentAgentVal').textContent = 'RL';
            document.getElementById('stepsVal').textContent = agent.steps;
            document.getElementById('reachedVal').textContent = agent.reached ? '✅' : '❌';
            document.getElementById('epsilonVal').textContent =
                agent.epsilon !== undefined ? agent.epsilon.toFixed(3) : '—';
        }
    }
}

// ─── Чтение настроек из UI ───────────────────────────────────────────────────

function readConfig() {
    config.populationSize = parseInt(document.getElementById('populationSize').value, 10);
    config.mutationRate   = parseFloat(document.getElementById('mutationRate').value);
    config.hiddenSize     = parseInt(document.getElementById('hiddenSize').value, 10);
    config.maxSteps       = parseInt(document.getElementById('maxSteps').value, 10);

    document.getElementById('populationSizeVal').textContent = config.populationSize;
    document.getElementById('mutationRateVal').textContent   = config.mutationRate.toFixed(2);
    document.getElementById('hiddenSizeVal').textContent     = config.hiddenSize;
    document.getElementById('maxStepsVal').textContent       = config.maxSteps;
}

// ─── Обработчики UI ──────────────────────────────────────────────────────────

function startTraining() {
    if (state.isTraining) return;
    state.isTraining = true;
    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnPause').disabled = false;
    update();
}

function pauseTraining() {
    state.isTraining = false;
    if (state.animFrameId) cancelAnimationFrame(state.animFrameId);
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnPause').disabled = true;
}

function resetTraining() {
    pauseTraining();
    chart.reset();
    init();
}

function toggleSpeed() {
    const speeds = [1, 5, 20];
    const idx = speeds.indexOf(state.speed);
    state.speed = speeds[(idx + 1) % speeds.length];
    document.getElementById('btnSpeed').textContent = `Скорость: ${state.speed}x`;
}

function changeMode(mode) {
    pauseTraining();
    state.mode = mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));

    // Показать/скрыть поле epsilon
    const epsilonRow = document.getElementById('epsilonRow');
    if (epsilonRow) epsilonRow.style.display = mode === 'rl' ? 'flex' : 'none';

    // Показать/скрыть панель архитектуры нейросети
    const nnArchPanel = document.getElementById('nnArchPanel');
    if (nnArchPanel) nnArchPanel.style.display = mode === 'rl' ? 'block' : 'none';

    resetTraining();
}

function changeMazeDifficulty() {
    pauseTraining();
    chart.reset();
    init();
}

function saveModel() {
    const agent = state.bestAgent || getCurrentAgent();
    if (!agent) return;
    const ok = Storage.saveModel(agent, { generation: state.generation, fitness: state.bestFitness });
    showToast(ok ? '✅ Модель сохранена' : '❌ Ошибка сохранения');
}

function loadModel() {
    const data = Storage.loadModel();
    if (!data) {
        showToast('❌ Нет сохранённой модели');
        return;
    }
    try {
        const brain = NeuralNetwork.deserialize(data.weights);
        const agent = new Agent(brain, brain.hiddenSize);
        agent.setEnvironment(state.maze, state.start, state.goal);
        if (state.mode === 'genetic') {
            state.population[0] = agent;
        } else {
            state.rlAgent.brain = brain;
        }
        showToast(`✅ Загружено (поколение ${data.generation})`);
    } catch (e) {
        showToast('❌ Ошибка загрузки');
    }
}

function downloadModel() {
    const agent = state.bestAgent || getCurrentAgent();
    if (!agent) return;
    Storage.downloadModel(agent, { generation: state.generation, fitness: state.bestFitness });
}

function uploadModel() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async e => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const data = await Storage.uploadModel(file);
            const brain = NeuralNetwork.deserialize(data.weights);
            const agent = new Agent(brain, brain.hiddenSize);
            agent.setEnvironment(state.maze, state.start, state.goal);
            if (state.mode === 'genetic') {
                state.population[0] = agent;
            } else {
                state.rlAgent.brain = brain;
            }
            showToast(`✅ Файл загружен (поколение ${data.generation})`);
        } catch (err) {
            showToast('❌ ' + err.message);
        }
    };
    input.click();
}

function showOptimalPath() {
    state.showOptimal = !state.showOptimal;
    document.getElementById('btnOptimal').textContent =
        state.showOptimal ? 'Скрыть A* путь' : 'Показать A* путь';
    renderFrame();
}

function exportStatistics() {
    chart.exportCSV();
}

function toggleSensors() {
    config.showSensors = !config.showSensors;
    document.getElementById('btnSensors').textContent =
        config.showSensors ? 'Скрыть сенсоры' : 'Показать сенсоры';
    renderFrame();
}

function showToast(msg) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 2500);
}

// ─── Навешивание обработчиков ────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    init();

    document.getElementById('btnStart').addEventListener('click', startTraining);
    document.getElementById('btnPause').addEventListener('click', pauseTraining);
    document.getElementById('btnReset').addEventListener('click', resetTraining);
    document.getElementById('btnSpeed').addEventListener('click', toggleSpeed);
    document.getElementById('btnOptimal').addEventListener('click', showOptimalPath);
    document.getElementById('btnSensors').addEventListener('click', toggleSensors);
    document.getElementById('btnSave').addEventListener('click', saveModel);
    document.getElementById('btnLoad').addEventListener('click', loadModel);
    document.getElementById('btnDownload').addEventListener('click', downloadModel);
    document.getElementById('btnUpload').addEventListener('click', uploadModel);
    document.getElementById('btnExport').addEventListener('click', exportStatistics);

    // Режим
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => changeMode(btn.dataset.mode));
    });

    // Сложность лабиринта
    document.getElementById('mazeDifficulty').addEventListener('change', changeMazeDifficulty);

    // Ползунки настроек
    ['populationSize', 'mutationRate', 'hiddenSize', 'maxSteps'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', () => {
                readConfig();
            });
        }
    });

    // Learning rate ползунок (RL)
    const lrEl = document.getElementById('learningRate');
    if (lrEl) {
        lrEl.addEventListener('input', () => {
            const val = parseFloat(lrEl.value).toFixed(4);
            const lrValEl = document.getElementById('lrValue');
            if (lrValEl) lrValEl.textContent = val;
        });
    }
});
