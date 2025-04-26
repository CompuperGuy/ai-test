#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VOCAB_SIZE 10
#define EMBED_SIZE 8
#define SEQ_LEN 4

// Random float between -1 and 1
float rand_float() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Dot product
float dot(float *a, float *b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Softmax function
void softmax(float *input, float *output, int size) {
    float max = input[0];
    for (int i = 1; i < size; i++) if (input[i] > max) max = input[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

int main() {
    srand(42);

    // Embedding table
    float embeddings[VOCAB_SIZE][EMBED_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < EMBED_SIZE; j++) {
            embeddings[i][j] = rand_float();
        }
    }

    // Example input tokens: (indexes into vocab)
    int input_tokens[SEQ_LEN] = {1, 2, 3, 4};

    // Embed input tokens
    float embedded[SEQ_LEN][EMBED_SIZE];
    for (int i = 0; i < SEQ_LEN; i++) {
        memcpy(embedded[i], embeddings[input_tokens[i]], sizeof(float) * EMBED_SIZE);
    }

    // Attention weights (very simple self-attention, dot product based)
    float attn_scores[SEQ_LEN][SEQ_LEN];
    float attn_probs[SEQ_LEN][SEQ_LEN];

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            attn_scores[i][j] = dot(embedded[i], embedded[j], EMBED_SIZE);
        }
        softmax(attn_scores[i], attn_probs[i], SEQ_LEN);
    }

    // Attention output
    float attended[SEQ_LEN][EMBED_SIZE] = {0};
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            for (int k = 0; k < EMBED_SIZE; k++) {
                attended[i][k] += attn_probs[i][j] * embedded[j][k];
            }
        }
    }

    // Feedforward (not shown fully here, but simple linear layers)
    // For now let's just print the attended outputs
    printf("Attended outputs:\n");
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_SIZE; j++) {
            printf("%f ", attended[i][j]);
        }
        printf("\n");
    }

    return 0;
}
