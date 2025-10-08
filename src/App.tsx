import { useEffect, useMemo, useState } from 'react';
import DrawingPad from './components/DrawingPad';
import MatrixGrid from './components/MatrixGrid';
import {
  HANDCRAFTED_KERNELS,
  KernelInfo,
  Matrix,
  convolve,
  createMatrix,
  dotProduct,
  flattenMatrix,
  flattenMatrices,
  maxPool,
  normalizeMatrix,
  normalizeVector,
  relu,
} from './cnn';
import { DIGIT_TEMPLATES } from './templates';
import './App.css';

type FeatureBundle = {
  kernel: KernelInfo;
  convolution: Matrix;
  activation: Matrix;
  pooled: Matrix;
};

type TemplateFeatureSet = {
  digit: number;
  normalizedMatrix: Matrix;
  featureVector: number[];
  pixelVector: number[];
};

type Prediction = {
  digit: number;
  score: number;
  convScore: number;
  pixelScore: number;
  probability: number;
};

const MATRIX_SIZE = 28;
const TARGET_SIZE = 20;
const CENTER = (MATRIX_SIZE - 1) / 2;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function bilinearSample(matrix: Matrix, y: number, x: number): number {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const clampedY = clamp(y, 0, rows - 1);
  const clampedX = clamp(x, 0, cols - 1);
  const y0 = Math.floor(clampedY);
  const x0 = Math.floor(clampedX);
  const y1 = Math.min(rows - 1, y0 + 1);
  const x1 = Math.min(cols - 1, x0 + 1);
  const wy = clampedY - y0;
  const wx = clampedX - x0;

  const v00 = matrix[y0][x0];
  const v01 = matrix[y0][x1];
  const v10 = matrix[y1][x0];
  const v11 = matrix[y1][x1];

  const top = v00 * (1 - wx) + v01 * wx;
  const bottom = v10 * (1 - wx) + v11 * wx;
  return top * (1 - wy) + bottom * wy;
}

function smoothMatrix(matrix: Matrix, iterations = 1): Matrix {
  let current = matrix;
  for (let iter = 0; iter < iterations; iter += 1) {
    const rows = current.length;
    const cols = current[0].length;
    const next = createMatrix(rows, cols, 0);

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        let total = 0;
        let count = 0;

        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const rr = row + dy;
            const cc = col + dx;
            if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
              total += current[rr][cc];
              count += 1;
            }
          }
        }
        next[row][col] = count ? total / count : 0;
      }
    }
    current = next;
  }
  return current;
}

function normalizeToUnit(matrix: Matrix): Matrix {
  let max = 0;
  for (const row of matrix) {
    for (const value of row) {
      if (value > max) {
        max = value;
      }
    }
  }
  if (max === 0) {
    return matrix.map((row) => row.map(() => 0));
  }
  return matrix.map((row) => row.map((value) => value / max));
}

function computeCenterOfMass(matrix: Matrix): { row: number; col: number } | null {
  let total = 0;
  let weightedRow = 0;
  let weightedCol = 0;

  for (let row = 0; row < matrix.length; row += 1) {
    for (let col = 0; col < matrix[row].length; col += 1) {
      const value = matrix[row][col];
      total += value;
      weightedRow += value * row;
      weightedCol += value * col;
    }
  }

  if (total === 0) {
    return null;
  }

  return {
    row: weightedRow / total,
    col: weightedCol / total,
  };
}

function translateMatrix(matrix: Matrix, shiftRow: number, shiftCol: number): Matrix {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const translated = createMatrix(rows, cols, 0);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const sourceRow = row - shiftRow;
      const sourceCol = col - shiftCol;
      if (sourceRow >= 0 && sourceRow < rows && sourceCol >= 0 && sourceCol < cols) {
        translated[row][col] = matrix[sourceRow][sourceCol];
      }
    }
  }

  return translated;
}

function normalizeDigitMatrix(matrix: Matrix): Matrix {
  if (!matrix.length || !matrix[0].length) {
    return createMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
  }

  let minRow = MATRIX_SIZE;
  let maxRow = -1;
  let minCol = MATRIX_SIZE;
  let maxCol = -1;
  const threshold = 0.05;

  for (let row = 0; row < MATRIX_SIZE; row += 1) {
    for (let col = 0; col < MATRIX_SIZE; col += 1) {
      if (matrix[row][col] > threshold) {
        if (row < minRow) minRow = row;
        if (row > maxRow) maxRow = row;
        if (col < minCol) minCol = col;
        if (col > maxCol) maxCol = col;
      }
    }
  }

  if (maxRow === -1 || maxCol === -1) {
    return createMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
  }

  const width = maxCol - minCol + 1;
  const height = maxRow - minRow + 1;
  const scale = TARGET_SIZE / Math.max(width, height);
  const scaledWidth = Math.max(1, Math.round(width * scale));
  const scaledHeight = Math.max(1, Math.round(height * scale));
  const offsetRow = Math.floor((MATRIX_SIZE - scaledHeight) / 2);
  const offsetCol = Math.floor((MATRIX_SIZE - scaledWidth) / 2);

  const scaled = createMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
  for (let row = 0; row < scaledHeight; row += 1) {
    for (let col = 0; col < scaledWidth; col += 1) {
      const sourceRow = minRow + row / scale;
      const sourceCol = minCol + col / scale;
      scaled[offsetRow + row][offsetCol + col] = bilinearSample(matrix, sourceRow, sourceCol);
    }
  }

  const smoothed = smoothMatrix(scaled, 1);
  const centered = (() => {
    const center = computeCenterOfMass(smoothed);
    if (!center) {
      return smoothed;
    }
    const shiftRow = Math.round(CENTER - center.row);
    const shiftCol = Math.round(CENTER - center.col);
    return translateMatrix(smoothed, shiftRow, shiftCol);
  })();

  return normalizeToUnit(centered);
}

function computeFeatureBundles(image: Matrix): FeatureBundle[] {
  return HANDCRAFTED_KERNELS.map((kernel) => {
    const convolution = convolve(image, kernel.matrix);
    const activation = relu(convolution);
    const pooled = maxPool(activation, 2);
    return { kernel, convolution, activation, pooled };
  });
}

function computeTemplateFeatureSets(): TemplateFeatureSet[] {
  return DIGIT_TEMPLATES.map((template) => {
    const normalizedMatrix = normalizeDigitMatrix(template.matrix);
    const bundles = computeFeatureBundles(normalizedMatrix);
    const pooled = bundles.map((bundle) => bundle.pooled);
    const featureVector = normalizeVector(flattenMatrices(pooled));
    const pixelVector = normalizeVector(flattenMatrix(normalizedMatrix));
    return {
      digit: template.digit,
      normalizedMatrix,
      featureVector,
      pixelVector,
    };
  });
}

function computePredictions(
  featureVector: number[],
  pixelVector: number[],
  templates: TemplateFeatureSet[],
): Prediction[] {
  if (!templates.length) {
    return [];
  }

  const rawPredictions = templates.map((template) => {
    const convScore = dotProduct(featureVector, template.featureVector);
    const pixelScore = dotProduct(pixelVector, template.pixelVector);
    const score = convScore * 0.7 + pixelScore * 0.3;
    return {
      digit: template.digit,
      convScore,
      pixelScore,
      score,
    };
  });

  const expScores = rawPredictions.map((item) => Math.exp(item.score * 4));
  const sum = expScores.reduce((total, value) => total + value, 0);

  return rawPredictions
    .map((item, index) => ({
      ...item,
      probability: sum > 0 ? expScores[index] / sum : 0,
    }))
    .sort((a, b) => b.probability - a.probability);
}

function averageActivation(matrix: Matrix): number {
  if (!matrix.length || !matrix[0].length) {
    return 0;
  }
  let total = 0;
  let count = 0;
  for (const row of matrix) {
    for (const value of row) {
      total += value;
      count += 1;
    }
  }
  return count ? total / count : 0;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function App() {
  const [rawMatrix, setRawMatrix] = useState<Matrix>(() => createMatrix(MATRIX_SIZE, MATRIX_SIZE, 0));
  const [activeKernelIndex, setActiveKernelIndex] = useState(0);

  const templateFeatures = useMemo(() => computeTemplateFeatureSets(), []);
  const normalizedInput = useMemo(() => normalizeDigitMatrix(rawMatrix), [rawMatrix]);
  const featureBundles = useMemo(() => computeFeatureBundles(normalizedInput), [normalizedInput]);
  const featureDisplays = useMemo(
    () =>
      featureBundles.map((bundle) => ({
        kernel: bundle.kernel,
        convolution: normalizeMatrix(bundle.convolution),
        activation: normalizeMatrix(bundle.activation),
        pooled: normalizeMatrix(bundle.pooled),
        strength: averageActivation(bundle.activation),
      })),
    [featureBundles],
  );

  const pooledFeatures = useMemo(() => featureBundles.map((bundle) => bundle.pooled), [featureBundles]);
  const featureVector = useMemo(
    () => normalizeVector(flattenMatrices(pooledFeatures)),
    [pooledFeatures],
  );
  const pixelVector = useMemo(
    () => normalizeVector(flattenMatrix(normalizedInput)),
    [normalizedInput],
  );
  const predictions = useMemo(
    () => computePredictions(featureVector, pixelVector, templateFeatures),
    [featureVector, pixelVector, templateFeatures],
  );
  useEffect(() => {
    setActiveKernelIndex((current) => {
      if (!featureDisplays.length) {
        return 0;
      }
      return Math.min(current, featureDisplays.length - 1);
    });
  }, [featureDisplays.length]);

  const goToPreviousKernel = () => {
    setActiveKernelIndex((index) => Math.max(0, index - 1));
  };

  const goToNextKernel = () => {
    setActiveKernelIndex((index) => Math.min(featureDisplays.length - 1, index + 1));
  };

  const activeFeature = featureDisplays[activeKernelIndex];

  const topPrediction = predictions[0];
  const referenceTemplate = topPrediction
    ? templateFeatures.find((item) => item.digit === topPrediction.digit)
    : undefined;
  const hasStroke = featureVector.some((value) => value > 0);

  return (
    <div className="app">
      <header className="hero">
        <h1>Как сверточная сеть распознает твою цифру</h1>
        <p>
          Нарисуй цифру, а дальше следи, как она проходит через свертки и pooling. Мы покажем тепловые карты,
          объясним силу каждого фильтра и дадим итоговый ответ сети.
        </p>
      </header>

      <main className="content">
        <section className="panel">
          <div className="panel__header">
            <h2>1. Нарисуй цифру</h2>
            <p>Белый фон — фон тетрадки, темные мазки — то, что сеть считает «чернилами».</p>
          </div>
          <div className="drawing-area">
            <DrawingPad onMatrixChange={setRawMatrix} />
            <div className="drawing-preview">
              <MatrixGrid
                matrix={normalizeMatrix(rawMatrix)}
                title="Сырые пиксели (28×28)"
                variant="large"
              />
              <MatrixGrid
                matrix={normalizeMatrix(normalizedInput)}
                title="После нормализации"
                variant="large"
              />
              <p className="preview-note">
                Мы масштабируем и центрируем цифру, чтобы линии попадали в ту же область, где сеть видела
                примеры при обучении.
              </p>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel__header">
            <h2>2. Что видит каждый фильтр</h2>
            <p>
              Мы рассмотрим фильтры по очереди: каждый ищет свой тип штриха и показывает реакцию после трёх шагов
              обработки.
            </p>
          </div>
          <div className="kernel-stepper">
            <div className="kernel-stepper__controls">
              <div className="kernel-stepper__progress">
                Шаг {featureDisplays.length ? activeKernelIndex + 1 : 0} из {featureDisplays.length}
              </div>
              <div className="kernel-stepper__buttons">
                <button
                  type="button"
                  className="kernel-stepper__button"
                  onClick={goToPreviousKernel}
                  disabled={activeKernelIndex === 0}
                >
                  Назад
                </button>
                <button
                  type="button"
                  className="kernel-stepper__button"
                  onClick={goToNextKernel}
                  disabled={activeKernelIndex >= featureDisplays.length - 1}
                >
                  Вперёд
                </button>
              </div>
            </div>
            {activeFeature ? (
              <article key={activeFeature.kernel.id} className="kernel-card kernel-card--single">
                <header className="kernel-card__header">
                  <div>
                    <span className="kernel-card__subtitle">Фильтр {activeKernelIndex + 1}</span>
                    <h3>{activeFeature.kernel.label}</h3>
                  </div>
                  <span className="kernel-card__strength">
                    сила отклика: {formatPercent(clamp(activeFeature.strength, 0, 1))}
                  </span>
                </header>
                <p className="kernel-card__explanation">{activeFeature.kernel.studentExplanation}</p>
                <p className="kernel-card__note">{activeFeature.kernel.description}</p>
                <div className="kernel-card__maps">
                  <MatrixGrid matrix={activeFeature.convolution} title="Шаг 1: свёртка" variant="small" />
                  <MatrixGrid matrix={activeFeature.activation} title="Шаг 2: ReLU" variant="small" />
                  <MatrixGrid matrix={activeFeature.pooled} title="Шаг 3: max-pooling" variant="small" />
                </div>
              </article>
            ) : (
              <p className="kernel-stepper__empty">
                Нарисуй цифру слева: как только появятся штрихи, мы начнём показ фильтров.
              </p>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel__header">
            <h2>3. Итоговое решение</h2>
            <p>Мы сравниваем признаки твоей цифры с эталонами, которые сеть запомнила для каждого класса.</p>
          </div>
          <div className="prediction-block">
            <div className="prediction-summary">
              {hasStroke && topPrediction ? (
                <>
                  <p className="prediction-answer">
                    Сеть уверена, что это <span className="prediction-answer__digit">{topPrediction.digit}</span>
                    <span className="prediction-answer__confidence">
                      {formatPercent(topPrediction.probability)}
                    </span>
                  </p>
                  <p className="prediction-detail">
                    Косинусное сходство по картам признаков: {topPrediction.convScore.toFixed(2)}, по пикселям:{' '}
                    {topPrediction.pixelScore.toFixed(2)}.
                  </p>
                </>
              ) : (
                <p className="prediction-placeholder">
                  Начни с пары мазков — пока сеть не видит штрихи, у нее нет ответа.
                </p>
              )}
            </div>
            <div className="prediction-bars">
              <ul>
                {predictions.slice(0, 5).map((prediction) => (
                  <li key={prediction.digit} className="prediction-bars__item">
                    <span className="prediction-bars__digit">{prediction.digit}</span>
                    <div className="prediction-bars__bar">
                      <div
                        className="prediction-bars__fill"
                        style={{ width: `${prediction.probability * 100}%` }}
                      />
                    </div>
                    <span className="prediction-bars__percent">
                      {formatPercent(prediction.probability)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            {referenceTemplate ? (
              <div className="prediction-reference">
                <MatrixGrid
                  matrix={normalizeMatrix(referenceTemplate.normalizedMatrix)}
                  title={`Эталон для «${referenceTemplate.digit}»`}
                  variant="medium"
                />
                <p>
                  Эталонная цифра — это усредненный образ, через тот же набор фильтров. Чем ближе твои карты к
                  нему, тем выше оценка.
                </p>
              </div>
            ) : null}
          </div>
        </section>

        <section className="panel panel--notes">
          <h2>Что обсудить с классом</h2>
          <ul>
            <li>Почему фильтр вертикальных линий реагирует на «1», но почти молчит на «0»?</li>
            <li>Как max-pooling уменьшает картинку и при этом сохраняет информацию о штрихах?</li>
            <li>Что будет, если цифру нарисовать очень маленькой или сместить в угол?</li>
          </ul>
        </section>
      </main>
    </div>
  );
}

export default App;

