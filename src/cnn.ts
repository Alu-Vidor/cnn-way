export type Matrix = number[][];

export type KernelInfo = {
  id: string;
  label: string;
  description: string;
  studentExplanation: string;
  matrix: Matrix;
};

export const HANDCRAFTED_KERNELS: KernelInfo[] = [
  {
    id: 'vertical-edge',
    label: 'Вертикальный фильтр',
    description: 'Находит высокие вертикальные штрихи: палочку у «1» или стенки у «0».',
    studentExplanation:
      'Представь линейку, которую мы держим вертикально. Если под ней есть тёмная полоска, фильтр загорается.',
    matrix: [
      [1, 0, -1],
      [1, 0, -1],
      [1, 0, -1],
    ],
  },
  {
    id: 'horizontal-edge',
    label: 'Горизонтальный фильтр',
    description: 'Замечает горизонтальные перекладины, важные для «2», «3», «4» или «7».',
    studentExplanation:
      'Этот фильтр словно ищет полочки и крышечки: если в цифре есть полоска поперёк, он сообщает об этом.',
    matrix: [
      [1, 1, 1],
      [0, 0, 0],
      [-1, -1, -1],
    ],
  },
  {
    id: 'main-diagonal',
    label: 'Главная диагональ',
    description: 'Ищет линии сверху слева вниз направо: такие штрихи есть у «2», «3», «7» и «9».',
    studentExplanation:
      'Подумай о наклонной линии, которая спускается, как горка. Когда цифра содержит такую линию, фильтр видит её.',
    matrix: [
      [0, 1, 1],
      [-1, 0, 1],
      [-1, -1, 0],
    ],
  },
  {
    id: 'anti-diagonal',
    label: 'Обратная диагональ',
    description: 'Реагирует на линии снизу слева вверх направо, например у «4» и «9».',
    studentExplanation:
      'Это как наклонная лестница, которая поднимается в гору. Если цифра поднимает линию вверх вправо, фильтр это замечает.',
    matrix: [
      [1, 1, 0],
      [1, 0, -1],
      [0, -1, -1],
    ],
  },
  {
    id: 'stroke-enhancer',
    label: 'Усилитель штрихов',
    description: 'Делает жирные штрихи ярче, а фон тише, чтобы выделить цифру.',
    studentExplanation:
      'Представь маркер, который подчёркивает самые яркие части твоего рисунка и стирает лишние пятна вокруг.',
    matrix: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  {
    id: 'blob-detector',
    label: 'Фильтр пятен',
    description: 'Любит круглые пятна и петельки — например, «животик» у «8» или «0».',
    studentExplanation:
      'Он смотрит на большие закрашенные области. Чем больше залитая часть, тем сильнее реакция фильтра.',
    matrix: [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
  },
];

export function createMatrix(rows: number, cols: number, fill = 0): Matrix {
  return Array.from({ length: rows }, () => Array(cols).fill(fill));
}

export function cloneMatrix(source: Matrix): Matrix {
  return source.map((row) => [...row]);
}

export function convolve(image: Matrix, kernel: Matrix): Matrix {
  const kernelSize = kernel.length;
  const resultRows = image.length - kernelSize + 1;
  const resultCols = image[0].length - kernelSize + 1;
  const result = createMatrix(resultRows, resultCols, 0);

  for (let row = 0; row < resultRows; row += 1) {
    for (let col = 0; col < resultCols; col += 1) {
      let sum = 0;
      for (let i = 0; i < kernelSize; i += 1) {
        for (let j = 0; j < kernelSize; j += 1) {
          sum += image[row + i][col + j] * kernel[i][j];
        }
      }
      result[row][col] = sum;
    }
  }

  return result;
}

export function relu(matrix: Matrix): Matrix {
  return matrix.map((row) => row.map((value) => (value > 0 ? value : 0)));
}

export function maxPool(matrix: Matrix, window: number): Matrix {
  const rows = Math.floor(matrix.length / window);
  const cols = Math.floor(matrix[0].length / window);
  const pooled = createMatrix(rows, cols, 0);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      let max = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < window; i += 1) {
        for (let j = 0; j < window; j += 1) {
          const value = matrix[row * window + i][col * window + j];
          if (value > max) {
            max = value;
          }
        }
      }
      pooled[row][col] = max;
    }
  }

  return pooled;
}

export function flattenMatrix(matrix: Matrix): number[] {
  const flattened: number[] = [];
  matrix.forEach((row) => flattened.push(...row));
  return flattened;
}

export function flattenMatrices(matrices: Matrix[]): number[] {
  const flattened: number[] = [];
  matrices.forEach((matrix) => {
    matrix.forEach((row) => {
      flattened.push(...row);
    });
  });
  return flattened;
}

export function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

export function vectorNorm(values: number[]): number {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
}

export function normalizeMatrix(matrix: Matrix): Matrix {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (const row of matrix) {
    for (const value of row) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < 1e-6) {
    return matrix.map((row) => row.map(() => 0));
  }

  const range = max - min;
  return matrix.map((row) => row.map((value) => (value - min) / range));
}

export function normalizeVector(values: number[]): number[] {
  const norm = vectorNorm(values);
  if (norm === 0) {
    return values.map(() => 0);
  }
  return values.map((value) => value / norm);
}

