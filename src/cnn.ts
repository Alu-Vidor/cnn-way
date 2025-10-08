export type Matrix = number[][];

export type KernelInfo = {
  id: string;
  label: string;
  description: string;
  matrix: Matrix;
};

export const HANDCRAFTED_KERNELS: KernelInfo[] = [
  {
    id: 'vertical-edge',
    label: 'Vertical Edge Detector',
    description: 'Highlights tall strokes such as the body of a 1 or the sides of a 0.',
    matrix: [
      [1, 0, -1],
      [1, 0, -1],
      [1, 0, -1],
    ],
  },
  {
    id: 'horizontal-edge',
    label: 'Horizontal Edge Detector',
    description: 'Emphasises horizontal bars that digits like 2, 3, 4, or 7 rely on.',
    matrix: [
      [1, 1, 1],
      [0, 0, 0],
      [-1, -1, -1],
    ],
  },
  {
    id: 'main-diagonal',
    label: 'Main Diagonal Detector',
    description: 'Responds to strokes going from top-left to bottom-right, useful for 2, 3, 7, and 9.',
    matrix: [
      [0, 1, 1],
      [-1, 0, 1],
      [-1, -1, 0],
    ],
  },
  {
    id: 'anti-diagonal',
    label: 'Anti-Diagonal Detector',
    description: 'Catches strokes that go from bottom-left to top-right, common in 4 and 9.',
    matrix: [
      [1, 1, 0],
      [1, 0, -1],
      [0, -1, -1],
    ],
  },
  {
    id: 'stroke-enhancer',
    label: 'Stroke Enhancer',
    description: 'Boosts thick strokes and suppresses the background to isolate the digit.',
    matrix: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  {
    id: 'blob-detector',
    label: 'Blob Detector',
    description: 'Acts like a blur that keeps filled loops such as the belly of an 8 or 0.',
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

