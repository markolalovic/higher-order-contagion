/**************************************************************************************************
 * SC_mod.c
 *
 * Modified:
 * - kg = actual triangle count for a node is only incremented AFTER Triangle() succeeds
 * - Simplified backtracking logic for illegal and failed matches
 *
 * Seems to work:
 * $ clang SC_d2_mod.c -o SC_d2_mod -lm
 * $ ./SC_d2_mod
 * #> Initial total stubs (xaus): 5433
 * #> Stub matching done. Remaining stubs (xaus): 0. Consecutive fails at end (naus): 0
 *   * Main loop : for matching stubs finished
 *   * 5433 / 3 = 1811 : kgi generation and sum logic produced total number of stubs % 3 = 0 
 *   * remaining 0 : all the stubs were used 
 * 
 * Adapted from:
 *   * O.T. Courtney and G. Bianconi
 *   * "Generalized network structures: the configuration model and the canonical ensemble of simplicial complexes"
 *   * Phys. Rev. E 93, 062311 (2016)
 *   * http://dx.doi.org/10.1103/PhysRevE.93.062311
 *   * https://github.com/ginestrab/Ensembles-of-Simplicial-Complexes/blob/8c32d11a281f31813c8e0693cd010b07e2c823b3/SC_d2.c    
 **************************************************************************************************/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>

#define N 1000
#define m 2
#define gamma2 2.5
#define Avoid 1
#define NX 15 // NOTE: max consecutive fails before giving up on current stub set
#define figure 1

int *kgi,*kg,***tri;

/*************************************************************************************************/
int Choose(double x){
	int i1 = -1, i;
	for (i=0;i<N;i++){
		if (kgi[i] > 0) { x-=kgi[i]; }
		if (x<0){ i1=i; break; }
	}
    if (i1 == -1 && x >= 0) {
        for(i=0; i<N; ++i) if(kgi[i]>0) return i;
    }
	return(i1);
}
/*************************************************************************************************/
int Check(int i1, int i2, int i3){
	int in,c=0;
    if (kg[i1] > 0 && (tri[i1][0] == NULL || tri[i1][1] == NULL)) {
        // this state should not be reached if kg is managed carefully
        return 1;
    }
	for(in=0; in < kg[i1]; in++){
		if(((tri[i1][0][in]==i2)&&(tri[i1][1][in]==i3))||((tri[i1][0][in]==i3)&&(tri[i1][1][in]==i2))){
			c=1;
			break;
		}
	}
	return(c);
}
/*************************************************************************************************/
int Triangle(int i1, int i2,int i3){
    kg[i1]++; kg[i2]++; kg[i3]++;

	int* temp_tri_i1_0 = (int*)realloc(tri[i1][0], kg[i1] * sizeof(int));
	int* temp_tri_i1_1 = (int*)realloc(tri[i1][1], kg[i1] * sizeof(int));
	int* temp_tri_i2_0 = (int*)realloc(tri[i2][0], kg[i2] * sizeof(int));
	int* temp_tri_i2_1 = (int*)realloc(tri[i2][1], kg[i2] * sizeof(int));
	int* temp_tri_i3_0 = (int*)realloc(tri[i3][0], kg[i3] * sizeof(int));
	int* temp_tri_i3_1 = (int*)realloc(tri[i3][1], kg[i3] * sizeof(int));

    if (!temp_tri_i1_0 || !temp_tri_i1_1 || !temp_tri_i2_0 || !temp_tri_i2_1 || !temp_tri_i3_0 || !temp_tri_i3_1) {
        perror("Realloc failed in Triangle");
        // revert kg increments if realloc failed
        kg[i1]--; kg[i2]--; kg[i3]--;
        return 1;
    }

    tri[i1][0] = temp_tri_i1_0; tri[i1][1] = temp_tri_i1_1;
    tri[i2][0] = temp_tri_i2_0; tri[i2][1] = temp_tri_i2_1;
    tri[i3][0] = temp_tri_i3_0; tri[i3][1] = temp_tri_i3_1;

	tri[i1][0][kg[i1]-1]=i2;
	tri[i1][1][kg[i1]-1]=i3;
	tri[i2][0][kg[i2]-1]=i1;
	tri[i2][1][kg[i2]-1]=i3;
	tri[i3][0][kg[i3]-1]=i1;
	tri[i3][1][kg[i3]-1]=i2;
    return 0;
}
/*************************************************************************************************/

int main(int argc, char** argv){
	int i,j,i1,i2,i3,naus,*k,**a;
	double xaus, x;

	FILE *fp_edges;
	FILE *fp_triangles;

	fp_edges=fopen("SC_d2_edges.txt","w");
	srand48(time(NULL));

	kgi=(int*)calloc(N,sizeof(int));
	kg=(int*)calloc(N,sizeof(int));
	k=(int*)calloc(N,sizeof(int));
	a=(int**)calloc(N,sizeof(int*));
	tri=(int***)calloc(N,sizeof(int**));

    if (!kgi || !kg || !k || !a || !tri) { perror("Calloc failed for main arrays"); fclose(fp_edges); return 1;}

	for(i=0;i<N;i++){
		a[i]=(int*)calloc(N,sizeof(int));
        tri[i]=(int**)calloc(2,sizeof(int*));
        if (!a[i] || !tri[i]) { perror("Calloc failed in loop"); fclose(fp_edges); /* add more cleanup */ return 1; }
        tri[i][0]=NULL; tri[i][1]=NULL;
    }

	xaus=0; int init_attempts = 0; int max_init_attempts = 1000;
    do { /* kgi initialization loop */
        xaus = 0;
        for(i=0;i<N;i++){
            kgi[i]=(int)(m*pow(drand48(),-1./(gamma2-1.)));
            double max_kgi_node = (N>2)?(double)(N-1)*(N-2)*0.5:0.0;
            if (kgi[i] < m && m > 0) kgi[i] = m;
            if (kgi[i] > max_kgi_node) kgi[i] = (int)max_kgi_node;
            if (kgi[i] < 0) kgi[i] = 0;
            kg[i]=0; k[i]=0; for(j=0;j<N;j++){ a[i][j]=0; }
        }
        for(i=0;i<N;i++){ xaus+=kgi[i]; }
        init_attempts++;
    } while ((xaus < 3 || (long long)xaus % 3 != 0) && init_attempts < max_init_attempts);

    if (xaus < 3 || (long long)xaus % 3 != 0) { /* kgi sum adjustment */
        printf("Initial sum of kgi is %.0f. Attempting adjustment.\n", xaus);
        int remainder = (int)((long long)xaus % 3);
        if (remainder != 0) {
            for (int r_attempt = 0; r_attempt < remainder; r_attempt++) {
                int eligible_node = -1;
                for(int node_idx = N-1; node_idx >=0; --node_idx) if (kgi[node_idx] > m) { eligible_node = node_idx; break;}
                if(eligible_node == -1) for(int node_idx = N-1; node_idx >=0; --node_idx) if (kgi[node_idx] > 0) { eligible_node = node_idx; break;}
                if (eligible_node != -1) {kgi[eligible_node]--; xaus--;}
                else { printf("Cannot adjust kgi sum further.\n"); break;}
            }
        }
        if (xaus < 3 || (long long)xaus % 3 != 0) {
             printf("Error: Could not obtain valid sum of kgi! Final xaus: %.0f\n", xaus);
             fclose(fp_edges); /* Add full cleanup */ return 1;
        }
    }
    printf("Initial total stubs (xaus): %.0f\n", xaus);

	naus=0; // failure counter for the current set of three stubs
	while(xaus>=3){
        if (naus >= 1 + Avoid * NX) {
            printf("Max total backtracks reached, stopping.\n");
            break;
        }
		x=xaus*drand48(); i1=Choose(x); if(i1<0){naus++; if(Avoid==1)continue; else break;}
		kgi[i1]--; xaus--;

		x=xaus*drand48(); i2=Choose(x); if(i2<0){kgi[i1]++;xaus++; naus++; if(Avoid==1)continue; else break;}
		kgi[i2]--; xaus--;

        if (xaus < 1 && N>2) {
            kgi[i1]++; xaus++; kgi[i2]++; xaus++; naus++;
            if (Avoid == 1) continue; else break;
        }
		x=xaus*drand48(); i3=Choose(x); if(i3<0){kgi[i1]++;xaus++; kgi[i2]++;xaus++; naus++; if(Avoid==1)continue; else break;}
		kgi[i3]--; xaus--;

		if((i1!=i2)&&(i2!=i3)&&(i3!=i1)&&(Check(i1,i2,i3)==0)){
            if (Triangle(i1,i2,i3) == 0) { // triangle successful (kg was incremented inside)
			    a[i1][i2]=1; a[i2][i1]=1;
			    a[i1][i3]=1; a[i3][i1]=1;
			    a[i2][i3]=1; a[i3][i2]=1;
                naus = 0; // reset consecutive failure counter
            } else { // Triangle failed (realloc problem) -> kg was already decremented inside Triangle
                naus++;
                // Stubs must be put back to kgi and xaus
				kgi[i1]++; kgi[i2]++; kgi[i3]++;
                xaus+=3;
            }
		} else { // standard illegal match (not distinct or triangle exists)
			naus++;
			if(Avoid==1){ // put stubs back
				kgi[i1]++; kgi[i2]++; kgi[i3]++;
                xaus+=3;
			}
            // if Avoid==0, stubs are considered consumed (kgi and xaus remain decremented)
            // and kg was never incremented for this illegal set
		}
	}
    printf("Stub matching done. Remaining stubs (xaus): %.0f. Consecutive fails at end (naus): %d\n", xaus, naus);

	for (i=0;i<N;i++){
		for(j=i+1;j<N;j++){ if(a[i][j]>0){ k[i]++; k[j]++; } }
	}

	if (figure==1){
		for (i=0;i<N;i++){ for(j=i+1;j<N;j++){ if(a[i][j]==1){ fprintf(fp_edges,"%d %d\n",i,j); }}}
        fp_triangles = fopen("SC_d2_triangles.txt", "w");
        // else {
        for (i = 0; i < N; i++) {
            for (j = 0; j < kg[i]; j++) {
                if (tri[i] == NULL || tri[i][0] == NULL || tri[i][1] == NULL) continue;
                int n1 = i, n2 = tri[i][0][j], n3 = tri[i][1][j];
                if      (n1 < n2 && n2 < n3) fprintf(fp_triangles, "%d %d %d\n", n1, n2, n3);
                else if (n1 < n3 && n3 < n2) fprintf(fp_triangles, "%d %d %d\n", n1, n3, n2);
                else if (n2 < n1 && n1 < n3) fprintf(fp_triangles, "%d %d %d\n", n2, n1, n3);
                else if (n2 < n3 && n3 < n1) fprintf(fp_triangles, "%d %d %d\n", n2, n3, n1);
                else if (n3 < n1 && n1 < n2) fprintf(fp_triangles, "%d %d %d\n", n3, n1, n2);
                else if (n3 < n2 && n2 < n1) fprintf(fp_triangles, "%d %d %d\n", n3, n2, n1);
            }
        }
        fclose(fp_triangles);
        // }
	}
	fclose(fp_edges);

    for(i=0;i<N;i++){
        if (a[i]) free(a[i]);
        if(tri[i] != NULL) {
            if(tri[i][0] != NULL) free(tri[i][0]);
            if(tri[i][1] != NULL) free(tri[i][1]);
            free(tri[i]);
        }
    }
    if (a) free(a); if (kgi) free(kgi); if (kg) free(kg); if (k) free(k); if (tri) free(tri);
	return 0;
}