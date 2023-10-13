#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nabc.h>

FILE* nabout;

int main(int argc, char* argv[] )
{
    nabout = stdout;

    MOLECULE_T *m;

    m = getpdb( argv[2], "");

    readparm(m, argv[1]);

    PARMSTRUCT_T* prm = rdparm( argv[1] );
    int natm = prm->Natom;

    POINT_T* x = malloc(sizeof(POINT_T) * natm);
    setxyz_from_mol( &m, NULL, x );

    mm_options( "cut=15., ntpr=1, nsnb=99999, diel = C, dielc = 80.40" );

    // nothing frozen or constrained
    int* frozen = parseMaskString( "@ZZZ", prm, (double *)x, 2 );
    int* constrained = parseMaskString( "@ZZZ", prm, (double *)x, 2 );

    mme_init_sff( prm, frozen, constrained, NULL, NULL );

    int nm = nmode( (double *) x, prm->Nat3, mme2, 0, 0, 0.0, 0.0, 0 );
    printf("nmode returns %d\n", nm);
}