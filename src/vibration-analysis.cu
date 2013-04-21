#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

#include "network.cuh"

#define S_C 167 // Sample Count
#define TS_F 4  // Data point per time-series
#define DIRECTORY "data/"

FILE * openfile(char * filelocation) {
  FILE * fp = fopen(filelocation, "r");
  if (fp == NULL) {
    printf("Couldn't open file %s. Aborting.\n", filelocation);
    exit(0);
  }

  return fp;
}

void test_vibration_analysis(Network *n) {
  FILE *fp;
  int i;
  char filelocation[100];
  float vib_table[S_C][TS_F],
        vib_spindle[S_C][TS_F],
        AE_table[S_C][TS_F],
        AE_spindle[S_C][TS_F],
        smcDC[S_C][TS_F],
        
        DOC[S_C],
        feed[S_C],
        VB[S_C];

  Neuron *neu;
  Connection *con;
  int *rate;

  // Store data
  for (i=0; i<S_C; i++) {
    sprintf(filelocation, "%s/processed/vib_table-%d-energies.data", DIRECTORY, i);

    // Read vib_table data
    fp = openfile(filelocation);
    float ta, tb; // tmp variables

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    vib_table[i][0] = ta;
    vib_table[i][1] = tb;

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    vib_table[i][2] = ta;
    vib_table[i][3] = tb;

    fclose(fp);

    // Read vib_spindle
    sprintf(filelocation, "%s/processed/vib_spindle-%d-energies.data", DIRECTORY, i);

    // Read vib_table data
    fp = openfile(filelocation);

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    vib_spindle[i][0] = ta;
    vib_spindle[i][1] = tb;

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    vib_spindle[i][2] = ta;
    vib_spindle[i][3] = tb;

    fclose(fp);

    // Read AE_table
    sprintf(filelocation, "%s/processed/AE_table-%d-energies.data", DIRECTORY, i);

    // Read vib_table data
    fp = openfile(filelocation);

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    AE_table[i][0] = ta;
    AE_table[i][1] = tb;

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    AE_table[i][2] = ta;
    AE_table[i][3] = tb;

    fclose(fp);

    // Read AE_spindle
    sprintf(filelocation, "%s/processed/AE_spindle-%d-energies.data", DIRECTORY, i);

    // Read vib_table data
    fp = openfile(filelocation);

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    AE_spindle[i][0] = ta;
    AE_spindle[i][1] = tb;

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    AE_spindle[i][2] = ta;
    AE_spindle[i][3] = tb;

    fclose(fp);

    // Read smcDC
    sprintf(filelocation, "%s/processed/smcDC-%d-energies.data", DIRECTORY, i);

    // Read vib_table data
    fp = openfile(filelocation);

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    smcDC[i][0] = ta;
    smcDC[i][1] = tb;

    fscanf(fp, "%f, %f\n", &ta, &tb); 
    smcDC[i][2] = ta;
    smcDC[i][3] = tb;

    fclose(fp);

    // Read feed
    sprintf(filelocation, "%s/raw/feed-%d.data", DIRECTORY, i);
    fp = openfile(filelocation);
    fscanf(fp, "%f", &ta);
    feed[i] = ta;

     // Read DOC
    sprintf(filelocation, "%s/raw/DOC-%d.data", DIRECTORY, i);
    fp = openfile(filelocation);
    fscanf(fp, "%f", &ta);
    DOC[i] = ta;

     // Read FB
    sprintf(filelocation, "%s/raw/VB-%d.data", DIRECTORY, i);
    fp = openfile(filelocation);
    fscanf(fp, "%f", &ta);
    VB[i] = ta;
  }
  // Memorized all data

  // Perform initial sampling to figure out cluster centres
  for (i=0; i<S_C; i++) {
    // Reset inputs 
    neu = n->neurons(true);
    for (int xii=0; xii<22; xii++) {
      neu[xii].input = 0;
    }

    n->neurons(neu, true);

    for (int j=0; j<1000; j++) {
      if (j == 100) { // Allow time for network to stabilize
        neu = n->neurons(true);
        neu[0].input = vib_table[i][0];
        neu[1].input = vib_table[i][1];
        neu[2].input = vib_table[i][2];
        neu[3].input = vib_table[i][3];
        neu[4].input = vib_spindle[i][0];
        neu[5].input = vib_spindle[i][1];
        neu[6].input = vib_spindle[i][2];
        neu[7].input = vib_spindle[i][3];
        neu[8].input = AE_table[i][0];
        neu[9].input = AE_table[i][1];
        neu[10].input = AE_table[i][2];
        neu[11].input = AE_table[i][3];
        neu[12].input = AE_spindle[i][0];
        neu[13].input = AE_spindle[i][1];
        neu[14].input = AE_spindle[i][2];
        neu[15].input = AE_spindle[i][3];
        neu[16].input = smcDC[i][0];
        neu[17].input = smcDC[i][1];
        neu[18].input = smcDC[i][2];
        neu[19].input = smcDC[i][3];
        neu[20].input = feed[i];
        neu[21].input = DOC[i];

        for (int xii=0; xii<22; xii++) {
          //neu[xii].input = 100;
        }

        n->neurons(neu, true);
      }

      n->time_step();
      n->find_firing_neurons();
      n->update_current();
      n->update_potential();
    }

    rate = n->spiking_rate(true);

    printf("%d\n", rate[4]);
  }
}

int main() {
  Network *n = new Network(45, 506);
  n->build_connections(22,22,1);
  n->randomize_weights();

  test_vibration_analysis(n);
}
