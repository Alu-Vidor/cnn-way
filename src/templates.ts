import { Matrix, createMatrix } from './cnn';

type PatternMap = Record<number, string[]>;

const RAW_PATTERNS_14: PatternMap = {
  0: [
    '....######....',
    '..##########..',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '..##########..',
    '....######....',
  ],
  1: [
    '......##......',
    '.....###......',
    '....####......',
    '....####......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '....######....',
    '....######....',
  ],
  2: [
    '....######....',
    '..##########..',
    '.###......###.',
    '.###......###.',
    '........###...',
    '.......###....',
    '......###.....',
    '.....###......',
    '....###.......',
    '...###........',
    '..###.........',
    '.###..........',
    '.############.',
    '.############.',
  ],
  3: [
    '....######....',
    '..##########..',
    '.###......###.',
    '.###......###.',
    '........###...',
    '.......###....',
    '...########...',
    '...########...',
    '........###...',
    '........###...',
    '.###......###.',
    '.###......###.',
    '..##########..',
    '....######....',
  ],
  4: [
    '.......###....',
    '......####....',
    '.....#####....',
    '....##.###....',
    '...###.###....',
    '..###..###....',
    '.###...###....',
    '.############.',
    '.############.',
    '.......###....',
    '.......###....',
    '.......###....',
    '.......###....',
    '.......###....',
  ],
  5: [
    '.############.',
    '.############.',
    '.###..........',
    '.###..........',
    '.###..........',
    '.#########....',
    '.##########...',
    '.###......###.',
    '.........###..',
    '.........###..',
    '.###......###.',
    '.###......###.',
    '..##########..',
    '....######....',
  ],
  6: [
    '....######....',
    '..##########..',
    '.###......###.',
    '.###..........',
    '.###..........',
    '.#########....',
    '.##########...',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '..##########..',
    '....######....',
  ],
  7: [
    '.############.',
    '.############.',
    '.........###..',
    '........###...',
    '.......###....',
    '......###.....',
    '.....###......',
    '....###.......',
    '....###.......',
    '...###........',
    '...###........',
    '..###.........',
    '..###.........',
    '..###.........',
  ],
  8: [
    '....######....',
    '..##########..',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '..##########..',
    '..##########..',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '..##########..',
    '....######....',
  ],
  9: [
    '....######....',
    '..##########..',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '.###......###.',
    '..##########..',
    '....########..',
    '.........###..',
    '.........###..',
    '.........###..',
    '.........###..',
    '..#########...',
    '..######......',
  ],
};

function patternToBaseMatrix(pattern: string[]): Matrix {
  return pattern.map((row) => {
    return Array.from(row).map((symbol) => (symbol === '#' ? 1 : 0));
  });
}

function upscaleMatrix(matrix: Matrix, factor: number): Matrix {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const scaled = createMatrix(rows * factor, cols * factor, 0);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = matrix[row][col];
      for (let dy = 0; dy < factor; dy += 1) {
        for (let dx = 0; dx < factor; dx += 1) {
          scaled[row * factor + dy][col * factor + dx] = value;
        }
      }
    }
  }

  return scaled;
}

function smoothMatrix(matrix: Matrix, iterations = 1): Matrix {
  let current = matrix;
  for (let iter = 0; iter < iterations; iter += 1) {
    const rows = current.length;
    const cols = current[0].length;
    const smoothed = createMatrix(rows, cols, 0);

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
        smoothed[row][col] = total / count;
      }
    }

    current = smoothed;
  }
  return current;
}

function normalizeMatrixValues(matrix: Matrix): Matrix {
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

export type DigitTemplate = {
  digit: number;
  matrix: Matrix;
  pattern14: string[];
};

export const DIGIT_TEMPLATES: DigitTemplate[] = Object.entries(RAW_PATTERNS_14).map(
  ([digit, pattern]) => {
    const baseMatrix = patternToBaseMatrix(pattern);
    const upscale = upscaleMatrix(baseMatrix, 2);
    const smooth = smoothMatrix(upscale, 2);
    const normalized = normalizeMatrixValues(smooth);
    return {
      digit: Number(digit),
      matrix: normalized,
      pattern14: pattern,
    };
  },
);

