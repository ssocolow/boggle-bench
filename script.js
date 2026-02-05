// Boggle Eval - Main Script

async function loadGame(gameId) {
    const response = await fetch(`data/${gameId}/game.json`);
    return response.json();
}

async function loadModels(gameId) {
    // Fetch the models index file to get list of available models
    const indexResponse = await fetch(`data/${gameId}/models-real/index.json`);
    const modelFiles = await indexResponse.json();

    // Load all model files
    const models = await Promise.all(
        modelFiles.map(async (filename) => {
            const response = await fetch(`data/${gameId}/models-real/${filename}`);
            return response.json();
        })
    );

    return models;
}

async function loadStats(gameId) {
    // Try to load stats file with mean/stdev data
    try {
        const response = await fetch(`data/${gameId}/stats.json`);
        if (!response.ok) return null;
        return response.json();
    } catch {
        return null;
    }
}

function renderGrid(grid, container, errorPositions = []) {
    container.innerHTML = '';

    for (let row = 0; row < grid.length; row++) {
        for (let col = 0; col < grid[row].length; col++) {
            const cell = document.createElement('div');
            cell.className = 'boggle-cell';
            cell.textContent = grid[row][col];

            // Check if this position has an error
            const hasError = errorPositions.some(
                ([errRow, errCol]) => errRow === row && errCol === col
            );
            if (hasError) {
                cell.classList.add('cell-error');
            }

            container.appendChild(cell);
        }
    }
}

function renderCorrectGrid(grid) {
    const container = document.getElementById('correct-grid');
    renderGrid(grid, container);
}

function compareGrids(modelGrid, correctGrid) {
    const errors = [];
    for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 5; col++) {
            if (modelGrid[row][col] !== correctGrid[row][col]) {
                errors.push([row, col]);
            }
        }
    }
    return errors;
}

function renderScoreChart(models, maxScore, stats = null) {
    const container = document.getElementById('score-chart');
    container.innerHTML = '';

    // Create bar for max score
    const maxBar = createChartBar('Max Possible', maxScore, maxScore, 'max-score');
    container.appendChild(maxBar);

    if (stats && stats.models) {
        // Use stats data with error bars (already sorted by mean)
        stats.models.forEach(modelStats => {
            const bar = createChartBarWithError(
                modelStats.model,
                modelStats.mean,
                modelStats.stdev,
                maxScore,
                'model-score'
            );
            container.appendChild(bar);
        });
    } else {
        // Fallback: sort models by score descending
        const sortedModels = [...models].sort((a, b) => b.wordScore - a.wordScore);
        sortedModels.forEach(model => {
            const bar = createChartBar(model.model, model.wordScore, maxScore, 'model-score');
            container.appendChild(bar);
        });
    }
}

function createChartBar(label, score, maxScore, type) {
    const container = document.createElement('div');
    container.className = 'chart-bar-container';

    const labelEl = document.createElement('span');
    labelEl.className = 'chart-label';
    labelEl.textContent = label;
    container.appendChild(labelEl);

    const barWrapper = document.createElement('div');
    barWrapper.className = 'chart-bar-wrapper';

    const bar = document.createElement('div');
    bar.className = `chart-bar ${type}`;
    const percentage = (score / maxScore) * 100;
    bar.style.width = `${percentage}%`;

    const valueEl = document.createElement('span');
    valueEl.className = 'chart-bar-value';
    valueEl.textContent = score;
    bar.appendChild(valueEl);

    barWrapper.appendChild(bar);
    container.appendChild(barWrapper);

    return container;
}

function createChartBarWithError(label, mean, stdev, maxScore, type) {
    const container = document.createElement('div');
    container.className = 'chart-bar-container';

    const labelEl = document.createElement('span');
    labelEl.className = 'chart-label';
    labelEl.textContent = label;
    container.appendChild(labelEl);

    const barWrapper = document.createElement('div');
    barWrapper.className = 'chart-bar-wrapper';

    const bar = document.createElement('div');
    bar.className = `chart-bar ${type}`;
    const percentage = (mean / maxScore) * 100;
    bar.style.width = `${percentage}%`;

    barWrapper.appendChild(bar);

    // Add error bar if stdev > 0
    if (stdev > 0) {
        const errorBar = document.createElement('div');
        errorBar.className = 'chart-error-bar';

        const stdevPercent = (stdev / maxScore) * 100;
        const leftPos = Math.max(0, percentage - stdevPercent);
        const rightPos = Math.min(100, percentage + stdevPercent);

        errorBar.style.left = `${leftPos}%`;
        errorBar.style.width = `${rightPos - leftPos}%`;

        // Add whisker caps
        const leftCap = document.createElement('div');
        leftCap.className = 'chart-error-cap chart-error-cap-left';
        errorBar.appendChild(leftCap);

        const rightCap = document.createElement('div');
        rightCap.className = 'chart-error-cap chart-error-cap-right';
        errorBar.appendChild(rightCap);

        // Add value at end of error bar
        const valueEl = document.createElement('span');
        valueEl.className = 'chart-bar-value chart-bar-value-error';
        valueEl.textContent = Math.round(mean);
        errorBar.appendChild(valueEl);

        barWrapper.appendChild(errorBar);
    } else {
        // No error bar, put value in the bar itself
        const valueEl = document.createElement('span');
        valueEl.className = 'chart-bar-value';
        valueEl.textContent = Math.round(mean);
        bar.appendChild(valueEl);
    }

    container.appendChild(barWrapper);

    return container;
}

function getThermometerClass(score, bestScore) {
    const percentage = (score / bestScore) * 100;
    if (percentage >= 100) return 'score-best';
    if (percentage >= 80) return 'score-high';
    if (percentage >= 60) return 'score-medium';
    return 'score-low';
}

function getLastUpdatedText(models) {
    const dates = models.map(m => new Date(m.date));
    const mostRecent = new Date(Math.max(...dates));
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    mostRecent.setHours(0, 0, 0, 0);

    const diffTime = today - mostRecent;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return '(updated today)';
    if (diffDays === 1) return '(updated yesterday)';
    return `(updated ${diffDays} days ago)`;
}

function renderModelCard(modelData, correctGrid, bestScore) {
    const card = document.createElement('div');
    card.className = 'model-card';

    const title = document.createElement('h3');
    title.textContent = modelData.model;
    card.appendChild(title);

    // Content container with grid and words side by side
    const content = document.createElement('div');
    content.className = 'model-card-content';

    const gridContainer = document.createElement('div');
    gridContainer.className = 'boggle-grid';

    const errors = compareGrids(modelData.transcriptionGrid, correctGrid);
    renderGrid(modelData.transcriptionGrid, gridContainer, errors);
    content.appendChild(gridContainer);

    // Words found section
    const wordsSection = document.createElement('div');
    wordsSection.className = 'model-words-found';

    const wordsHeader = document.createElement('h4');
    wordsHeader.textContent = `Words Found (${modelData.wordsFound.length})`;
    wordsSection.appendChild(wordsHeader);

    const wordsList = document.createElement('div');
    wordsList.className = 'model-words-list';
    wordsList.textContent = modelData.wordsFound.sort().join(', ');
    wordsSection.appendChild(wordsList);

    // Mistaken words section
    if (modelData.mistakenWords && modelData.mistakenWords.length > 0) {
        const mistakenHeader = document.createElement('h4');
        mistakenHeader.className = 'mistaken-words-header';
        mistakenHeader.textContent = `Mistaken Words (${modelData.mistakenWords.length})`;
        wordsSection.appendChild(mistakenHeader);

        const mistakenList = document.createElement('div');
        mistakenList.className = 'model-words-list mistaken-words-list';
        mistakenList.textContent = modelData.mistakenWords.sort().join(', ');
        wordsSection.appendChild(mistakenList);
    }

    content.appendChild(wordsSection);
    card.appendChild(content);

    const stats = document.createElement('div');
    stats.className = 'model-stats';

    const correctCount = 25 - errors.length;
    stats.innerHTML = `
        <span class="model-stat">Transcription: <strong>${correctCount}/25</strong></span>
        <span class="model-stat">Word Score: <strong>${modelData.wordScore}</strong></span>
    `;
    card.appendChild(stats);

    // Add thermometer
    const thermometer = document.createElement('div');
    thermometer.className = 'thermometer-container';

    const thermometerLabel = document.createElement('div');
    thermometerLabel.className = 'thermometer-label';
    thermometerLabel.innerHTML = `<span>Score vs Best</span><span>${modelData.wordScore}/${bestScore}</span>`;
    thermometer.appendChild(thermometerLabel);

    const track = document.createElement('div');
    track.className = 'thermometer-track';

    const fill = document.createElement('div');
    fill.className = `thermometer-fill ${getThermometerClass(modelData.wordScore, bestScore)}`;
    const percentage = Math.min((modelData.wordScore / bestScore) * 100, 100);
    fill.style.width = `${percentage}%`;

    track.appendChild(fill);
    thermometer.appendChild(track);
    card.appendChild(thermometer);

    return card;
}

async function init() {
    const gameId = 'game1';

    try {
        // Load game data
        const gameData = await loadGame(gameId);

        // Set image
        document.getElementById('boggle-image').src = gameData.image;

        // Render correct grid
        renderCorrectGrid(gameData.correctGrid);

        // Set stats
        document.getElementById('max-score').textContent = gameData.maxScore;
        if (gameData.validWords) {
            document.getElementById('valid-words-count').textContent = gameData.validWords.length;
            // Populate valid words list
            const validWordsList = document.getElementById('valid-words-list');
            validWordsList.textContent = gameData.validWords.sort().join(', ');
        }

        // Load model results and stats in parallel
        const [models, stats] = await Promise.all([
            loadModels(gameId),
            loadStats(gameId)
        ]);

        // Find best model score
        const bestScore = Math.max(...models.map(m => m.wordScore));

        // Set last updated text
        document.getElementById('last-updated').textContent = getLastUpdatedText(models);

        // Set sample size if stats available
        if (stats && stats.models && stats.models.length > 0) {
            const minN = Math.min(...stats.models.map(m => m.n));
            document.getElementById('sample-size').textContent = `(n=${minN})`;
        }

        // Render score chart (with error bars if stats available)
        renderScoreChart(models, gameData.maxScore, stats);

        // Render model cards
        const modelCardsContainer = document.getElementById('model-cards');
        models.forEach(modelData => {
            const card = renderModelCard(modelData, gameData.correctGrid, bestScore);
            modelCardsContainer.appendChild(card);
        });

    } catch (error) {
        console.error('Error loading game data:', error);
    }
}

document.addEventListener('DOMContentLoaded', init);
