/**
 * storage.js — сохранение и загрузка моделей нейронной сети.
 *
 * Поддерживает:
 *  - localStorage (браузер)
 *  - скачивание JSON-файла
 *  - загрузку из файла
 *  - экспорт статистики в CSV
 */

const STORAGE_KEY = 'neuralMazeModel';
const STATS_KEY = 'neuralMazeStats';

export class Storage {
    /**
     * Сохранить модель агента в localStorage.
     * @param {import('./agent.js').Agent} agent
     * @param {{generation:number, fitness:number}} metadata
     */
    static saveModel(agent, metadata) {
        const data = {
            weights: agent.brain.serialize(),
            generation: metadata.generation,
            fitness: metadata.fitness,
            timestamp: Date.now(),
        };
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
            return true;
        } catch (e) {
            console.error('Ошибка сохранения модели:', e);
            return false;
        }
    }

    /**
     * Загрузить модель из localStorage.
     * @returns {{weights:object, generation:number, fitness:number, timestamp:number}|null}
     */
    static loadModel() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return null;
            return JSON.parse(raw);
        } catch (e) {
            console.error('Ошибка загрузки модели:', e);
            return null;
        }
    }

    /**
     * Скачать модель как JSON-файл.
     * @param {import('./agent.js').Agent} agent
     * @param {{generation:number, fitness:number}} metadata
     */
    static downloadModel(agent, metadata) {
        const data = {
            weights: agent.brain.serialize(),
            generation: metadata.generation,
            fitness: metadata.fitness,
            timestamp: Date.now(),
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neural-maze-gen${metadata.generation}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Загрузить модель из File (возвращает Promise с данными).
     * @param {File} file
     * @returns {Promise<object>}
     */
    static uploadModel(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => {
                try {
                    const data = JSON.parse(e.target.result);
                    resolve(data);
                } catch (err) {
                    reject(new Error('Неверный формат файла'));
                }
            };
            reader.onerror = () => reject(new Error('Ошибка чтения файла'));
            reader.readAsText(file);
        });
    }

    /**
     * Проверить наличие сохранённой модели.
     * @returns {boolean}
     */
    static hasModel() {
        return localStorage.getItem(STORAGE_KEY) !== null;
    }

    /**
     * Удалить сохранённую модель.
     */
    static deleteModel() {
        localStorage.removeItem(STORAGE_KEY);
    }

    /**
     * Сохранить статистику обучения в localStorage.
     * @param {object} chartData
     */
    static saveStatistics(chartData) {
        try {
            localStorage.setItem(STATS_KEY, JSON.stringify(chartData));
        } catch (e) {
            console.error('Ошибка сохранения статистики:', e);
        }
    }

    /**
     * Загрузить статистику обучения из localStorage.
     * @returns {object|null}
     */
    static loadStatistics() {
        try {
            const raw = localStorage.getItem(STATS_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            return null;
        }
    }

    /**
     * Экспортировать данные в CSV и скачать файл.
     * @param {object} data - объект с массивами по ключам
     * @param {string} [filename='training-stats.csv']
     */
    static exportToCSV(data, filename = 'training-stats.csv') {
        const keys = Object.keys(data);
        if (keys.length === 0) return;

        const maxLen = Math.max(...keys.map(k => data[k].length));
        const rows = [keys.join(',')];

        for (let i = 0; i < maxLen; i++) {
            rows.push(keys.map(k => data[k][i] ?? '').join(','));
        }

        const csv = rows.join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}
