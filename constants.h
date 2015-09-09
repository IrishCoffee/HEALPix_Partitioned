#ifndef HOST_CONSTANTS_H
#define HOST_CONSTANTS_H

const double pi=3.141592653589793238462643383279502884197;
const double twopi=6.283185307179586476925286766559005768394;
const double inv_twopi=1.0/twopi;
const double fourpi=12.56637061435917295385057353311801153679;
const double halfpi=1.570796326794896619231321691639751442099;
const double inv_halfpi=0.6366197723675813430755350534900574; // 1 / (pi / 2) = 1 / halfpi
const double inv_sqrt4pi = 0.2820947917738781434740397257803862929220;

const double onethird = 1.0 / 3.0;
const double twothird = 2.0 / 3.0;
const double fourthird = 4.0 / 3.0;

const int max_order = 13;
const int MAX_MATCH = 1;
const int MAX_RANGE_PAIR = 20; //10 pairs
const double eps = 1e-8;

const int block_size = 512;
const int TILE_SIZE = 1024;
struct PIX_NODE
{
	double ra; 
	double dec;
	int pix;
};
#endif
