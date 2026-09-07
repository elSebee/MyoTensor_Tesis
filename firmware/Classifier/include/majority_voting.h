#pragma once
/*
 * ============================================================
 *  majority_voting.h — Filtro Temporal por Votación Mayoritaria
 * ============================================================
 */

#include <cstdint>

template <int N = 5, int NUM_CLASSES = 4>
class MajorityVote {
private:
  int buffer[N] = {0};
  int idx = 0;

public:
  void reset() {
    for (int i = 0; i < N; i++) buffer[i] = 0;
    idx = 0;
  }

  int update(int new_prediction) {
    buffer[idx] = new_prediction;
    idx = (idx + 1) % N;

    int counts[NUM_CLASSES] = {0};
    for (int i = 0; i < N; i++) {
      if (buffer[i] >= 0 && buffer[i] < NUM_CLASSES) {
        counts[buffer[i]]++;
      }
    }

    int mode_class = 0;
    int max_count = -1;
    for (int c = 0; c < NUM_CLASSES; c++) {
      if (counts[c] > max_count) {
        max_count = counts[c];
        mode_class = c;
      }
    }
    return mode_class;
  }
};
