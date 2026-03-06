/**
 * chart.js — График обучения: отображает прогресс по поколениям.
 *
 * Рисует линии avg/max/min fitness и successRate.
 */
export class TrainingChart {
    /**
     * @param {HTMLCanvasElement} canvas
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        this.history = {
            generations: [],
            avgFitness: [],
            maxFitness: [],
            minFitness: [],
            stepsToGoal: [],
            successRate: [],
            maxStepsValues: [],
        };

        /** Отступы вокруг области графика */
        this.padding = { top: 20, right: 20, bottom: 40, left: 60 };
    }

    /**
     * Обновить историю по итогам поколения.
     * @param {number}   generation - номер поколения
     * @param {import('./agent.js').Agent[]} population
     */
    update(generation, population) {
        const fitnesses = population.map(a => a.fitness);
        const avg = fitnesses.reduce((s, f) => s + f, 0) / fitnesses.length;
        const max = Math.max(...fitnesses);
        const min = Math.min(...fitnesses);
        const reached = population.filter(a => a.reached);
        const successRate = reached.length / population.length;
        const avgSteps = reached.length > 0
            ? reached.reduce((s, a) => s + a.steps, 0) / reached.length
            : 0;

        this.history.generations.push(generation);
        this.history.avgFitness.push(avg);
        this.history.maxFitness.push(max);
        this.history.minFitness.push(min);
        this.history.stepsToGoal.push(avgSteps);
        this.history.successRate.push(successRate * 100);

        // Сохранение адаптивного maxSteps для RL агента
        if (population[0] && population[0].currentMaxSteps !== undefined) {
            this.history.maxStepsValues.push(population[0].currentMaxSteps);
        }

        this.draw();
    }

    /**
     * Нарисовать график.
     */
    draw() {
        const { canvas, ctx, history, padding: P } = this;
        const W = canvas.width;
        const H = canvas.height;

        ctx.clearRect(0, 0, W, H);

        // Фон
        ctx.fillStyle = 'rgba(15, 15, 40, 0.8)';
        ctx.fillRect(0, 0, W, H);

        const N = history.generations.length;
        if (N < 2) return;

        const plotW = W - P.left - P.right;
        const plotH = H - P.top - P.bottom;

        // Диапазон fitness
        const allValues = [...history.avgFitness, ...history.maxFitness, ...history.minFitness];
        const minVal = Math.min(...allValues);
        const maxVal = Math.max(...allValues);
        const range = maxVal - minVal || 1;

        /** Преобразовать значение в координату по Y */
        const toY = v => P.top + plotH - ((v - minVal) / range) * plotH;
        /** Преобразовать индекс в координату по X */
        const toX = i => P.left + (i / (N - 1)) * plotW;

        // Сетка
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 1;
        for (let g = 0; g <= 5; g++) {
            const y = P.top + (g / 5) * plotH;
            ctx.beginPath();
            ctx.moveTo(P.left, y);
            ctx.lineTo(P.left + plotW, y);
            ctx.stroke();
        }

        // Оси
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(P.left, P.top);
        ctx.lineTo(P.left, P.top + plotH);
        ctx.lineTo(P.left + plotW, P.top + plotH);
        ctx.stroke();

        // Метки Y
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let g = 0; g <= 5; g++) {
            const v = minVal + (1 - g / 5) * range;
            const y = P.top + (g / 5) * plotH;
            ctx.fillText(v.toFixed(0), P.left - 5, y + 4);
        }

        // Метки X
        ctx.textAlign = 'center';
        const step = Math.max(1, Math.floor(N / 5));
        for (let i = 0; i < N; i += step) {
            ctx.fillText(history.generations[i], toX(i), P.top + plotH + 15);
        }

        // Линии
        const lines = [
            { key: 'maxFitness', color: '#4ade80', label: 'Max' },
            { key: 'avgFitness', color: '#60a5fa', label: 'Avg' },
            { key: 'minFitness', color: '#f87171', label: 'Min' },
        ];

        for (const { key, color } of lines) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < N; i++) {
                const x = toX(i);
                const y = toY(history[key][i]);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Легенда
        const legendX = P.left + 10;
        let legendY = P.top + 10;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        for (const { color, label } of lines) {
            ctx.fillStyle = color;
            ctx.fillRect(legendX, legendY - 8, 14, 3);
            ctx.fillStyle = 'rgba(255,255,255,0.8)';
            ctx.fillText(label, legendX + 18, legendY);
            legendY += 16;
        }

        // Success rate (правая ось, зелёная)
        if (history.successRate.some(v => v > 0)) {
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            for (let i = 0; i < N; i++) {
                const x = toX(i);
                const y = P.top + plotH - (history.successRate[i] / 100) * plotH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.setLineDash([]);

            // Подпись
            ctx.fillStyle = '#fbbf24';
            ctx.fillRect(legendX, legendY - 8, 14, 3);
            ctx.fillStyle = 'rgba(255,255,255,0.8)';
            ctx.fillText('Success%', legendX + 18, legendY);
            legendY += 16;
        }

        // Линия адаптивного maxSteps (нормированная, серая пунктирная)
        if (history.maxStepsValues && history.maxStepsValues.length > 1) {
            const maxInData = Math.max(...history.maxStepsValues);
            ctx.strokeStyle = 'rgba(156, 163, 175, 0.5)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            for (let i = 0; i < Math.min(N, history.maxStepsValues.length); i++) {
                const x         = toX(i);
                const normalized = history.maxStepsValues[i] / maxInData;
                const y         = P.top + plotH - normalized * plotH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.setLineDash([]);

            // Подпись в легенде
            ctx.fillStyle = 'rgba(156, 163, 175, 0.5)';
            ctx.fillRect(legendX, legendY - 8, 14, 3);
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.fillText('MaxSteps', legendX + 18, legendY);
        }

        // Заголовок
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Обучение', W / 2, 12);
    }

    /**
     * Очистить canvas.
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Сбросить историю обучения.
     */
    reset() {
        this.history = {
            generations: [],
            avgFitness: [],
            maxFitness: [],
            minFitness: [],
            stepsToGoal: [],
            successRate: [],
            maxStepsValues: [],
        };
        this.clear();
    }

    /**
     * Экспортировать историю как CSV (скачивание файла).
     */
    exportCSV() {
        const { history: h } = this;
        const N = h.generations.length;
        const rows = ['generation,avgFitness,maxFitness,minFitness,stepsToGoal,successRate'];
        for (let i = 0; i < N; i++) {
            rows.push([
                h.generations[i],
                h.avgFitness[i]?.toFixed(2),
                h.maxFitness[i]?.toFixed(2),
                h.minFitness[i]?.toFixed(2),
                h.stepsToGoal[i]?.toFixed(2),
                h.successRate[i]?.toFixed(2),
            ].join(','));
        }
        const csv = rows.join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'training-stats.csv';
        a.click();
        URL.revokeObjectURL(url);
    }
}
