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
}

int main() {
  Network *n = new Network(45, 506);
  n->build_connections(22,22,1);
  n->randomize_weights();

  test_vibration_analysis(n);
}
