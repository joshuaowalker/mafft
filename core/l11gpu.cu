/*
 * CUDA implementation of l11gpu_align() (see l11gpu.h), built only with `make CUDA=1`.
 *
 * The same integer DP as Lfill_int() + Ltracking() and as the Metal kernel in l11gpu.metal: one
 * pair per 32-thread warp, each lane owning a contiguous run of C = ceil(l2/32) columns and
 * keeping their state (previous-row value, vertical-gap state and its start row, seq2 code) in
 * registers: the kernel is instantiated for every C, so the per-lane arrays have a
 * compile-time size and fully unrolled loops.  The horizontal-gap state
 * H_j = (j-1)*ext + max_{k<=j-2} (prev[k] - k*ext), with the first k attaining it, is a prefix
 * maximum: each lane scans its own columns and one warp-wide scan carries the maximum across
 * lanes.  The traceback matrix (int16, lane-interleaved so that a warp's stores are contiguous)
 * stays in device memory; lane 0 walks it as Ltracking() does and writes the two aligned
 * strings, so only results come back to the host.
 *
 * Integer arithmetic only, the same comparisons and the same tie rules, so the results do not
 * depend on the GPU.  Anything the kernel cannot take keeps status -1 and goes to the CPU.
 * MAFFT_NOGPU=1 disables it; no device, no driver or any CUDA error also leaves everything to
 * the CPU.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
extern "C" {
#include "l11gpu.h"
}

typedef struct { uint32_t s1off, s2off; int32_t l1, l2; uint32_t ijpoff, outoff; int32_t pad0, pad1; } PairDesc;
typedef struct { int32_t ncode, pen, ext, thr, gapc, npairs; } Params;
typedef struct { int32_t maxwm, endi, endj, off1, off2, start, status, pad; } Result;

#define CMIN 2      /* l2 >= 64 */
#define CMAXV 64    /* l2 <= 2048 */
#define WARPS_PER_BLOCK 4
#define SMTX_MAX 4096   /* ints of score matrix kept in shared memory: (ncode+1)^2 <= 4096 */
#define FULL 0xffffffffu

static __device__ __forceinline__ void comb( int &av, int &ak, int bv, int bk ) { if( bv > av ) { av = bv; ak = bk; } }

template <int C>
__global__ void __launch_bounds__( 32 * WARPS_PER_BLOCK )
l11k( const unsigned char * __restrict__ codes, const char * __restrict__ chars, const int * __restrict__ gmtx,
      const PairDesc * __restrict__ pd, Params P, short * __restrict__ ijpbuf, char * __restrict__ outbuf, Result * __restrict__ res )
{
	__shared__ int smtx[SMTX_MAX];
	const int nc1 = P.ncode + 1;
	const int *mtx = gmtx;
	if( nc1 * nc1 <= SMTX_MAX )
	{
		for( int x=threadIdx.x; x<nc1*nc1; x+=blockDim.x ) smtx[x] = gmtx[x];
		__syncthreads();
		mtx = smtx;
	}
	const int lane = threadIdx.x & 31;
	const int gid = blockIdx.x * ( blockDim.x >> 5 ) + ( threadIdx.x >> 5 );
	if( gid >= P.npairs ) return;   /* whole warps only */
	const PairDesc d = pd[gid];
	const int l1 = d.l1, l2 = d.l2;
	const int pen = P.pen, ext = P.ext, thr = P.thr;
	const int lstop = l1 + l2 + 1;
	const unsigned char *c1 = codes + d.s1off;
	const unsigned char *c2 = codes + d.s2off;
	short *ijp = ijpbuf + d.ijpoff;
	const int W = 32 * C;
	const int j0 = lane * C + 1;
	int Pv[C], VM[C], PK[C];
	const int u10 = c1[0];
#pragma unroll
	for( int t=0; t<C; t++ )
	{
		int j = j0 + t;
		int cj = ( j < l2 ) ? (int)c2[j] : P.ncode;
		int cjm = ( j <= l2 ) ? (int)c2[j-1] : P.ncode;
		Pv[t] = mtx[u10*nc1 + cj];
		VM[t] = mtx[u10*nc1 + cjm];
		PK[t] = cj;
	}
	int maxwm = INT_MIN, endi = 0, endj = 0;
	const int c20 = c2[0];
	for( int i=1; i<=l1; i++ )
	{
		const int prev0 = mtx[ c20*nc1 + (int)c1[i-1] ];
		const int *mrow = mtx + ( ( i < l1 ) ? (int)c1[i] : u10 ) * nc1;
		int last = 0, av = INT_MIN, ak = -1, a2v = INT_MIN, a2k = -1;
#pragma unroll
		for( int t=0; t<C; t++ )
		{
			int j = j0 + t;
			if( j <= l2 )
			{
				comb( av, ak, Pv[t] - j*ext, j );
				if( t <= C-2 ) { a2v = av; a2k = ak; }
			}
		}
		last = Pv[C-1];
		/* inclusive scan of lane aggregates (earlier lanes first) */
		int sv = av, sk = ak;
#pragma unroll
		for( int dd=1; dd<32; dd<<=1 )
		{
			int ov = __shfl_up_sync( FULL, sv, dd ), ok = __shfl_up_sync( FULL, sk, dd );
			if( lane >= dd ) { int nv = ov, nk = ok; comb( nv, nk, sv, sk ); sv = nv; sk = nk; }
		}
		int ev = __shfl_up_sync( FULL, sv, 1 ), ek = __shfl_up_sync( FULL, sk, 1 );
		{ int tv = prev0, tk = 0; if( lane > 0 ) comb( tv, tk, ev, ek ); ev = tv; ek = tk; }
		/* incl up to this lane's second-to-last column, passed to the next lane */
		int s2v = ev, s2k = ek; comb( s2v, s2k, a2v, a2k );
		int Sv = __shfl_up_sync( FULL, s2v, 1 ), Sk = __shfl_up_sync( FULL, s2k, 1 );
		int pl = __shfl_up_sync( FULL, last, 1 );
		if( lane == 0 ) { Sv = prev0; Sk = 0; pl = prev0; }
		int Rv = ev, Rk = ek, o1 = pl, o2 = 0;
		int lmax = INT_MIN, lj = INT_MAX;
		short *ijrow = ijp + (size_t)i*W + lane;
#pragma unroll
		for( int t=0; t<C; t++ )
		{
			int j = j0 + t;
			if( j <= l2 )
			{
				int hv, hk;
				if( t == 0 ) { hv = Sv; hk = Sk; }
				else if( t == 1 ) { hv = Rv; hk = Rk; }
				else { comb( Rv, Rk, o2 - ( j-2 )*ext, j-2 ); hv = Rv; hk = Rk; }
				int p = o1;
				int wm = p, ij = 0, g;
				g = hv + ( j-1 )*ext + pen;
				if( g > wm ) { wm = g; ij = -( j - hk ); }
				int vm = VM[t], vmp = PK[t] >> 8;
				g = vm + pen;
				if( g > wm ) { wm = g; ij = i - vmp; }
				if( p > vm ) { vm = p; vmp = i-1; }
				VM[t] = vm + ext;
				PK[t] = ( vmp << 8 ) | ( PK[t] & 255 );
				if( wm > lmax ) { lmax = wm; lj = j; }
				if( wm < thr ) { ij = lstop; wm = thr; }
				ijrow[t*32] = (short)ij;
				int old = Pv[t];
				Pv[t] = wm + mrow[ PK[t] & 255 ];
				o2 = o1; o1 = old;
			}
		}
#if __CUDA_ARCH__ >= 800
		int rmax = __reduce_max_sync( FULL, lmax );
		int rj = __reduce_min_sync( FULL, (unsigned)( lmax == rmax ? lj : INT_MAX ) );
#else
		int rmax = lmax;
		for( int dd=16; dd; dd>>=1 ) rmax = max( rmax, __shfl_xor_sync( FULL, rmax, dd ) );
		int rj = ( lmax == rmax ) ? lj : INT_MAX;
		for( int dd=16; dd; dd>>=1 ) rj = min( rj, __shfl_xor_sync( FULL, rj, dd ) );
#endif
		if( rmax > maxwm ) { maxwm = rmax; endi = i; endj = rj; }
	}
	__syncwarp();
	if( lane != 0 ) return;
	char *m1 = outbuf + d.outoff;
	char *m2 = m1 + ( l1 + l2 + 1 );
	const char *s1 = chars + d.s1off;
	const char *s2 = chars + d.s2off;
	const char gap = (char)P.gapc;
	int pos = l1 + l2;
	m1[pos] = 0; m2[pos] = 0;
	Result r; r.maxwm = maxwm; r.endi = endi; r.endj = endj; r.off1 = 0; r.off2 = 0; r.pad = 0;
	if( ijp[(size_t)endi*W + ((endj-1)%C)*32 + (endj-1)/C] == lstop ) { r.status = 1; r.start = pos; res[gid] = r; return; }
	int iin = endi, jin = endj, ifi = 0, jfi = 0, limk = l1 + l2, status = 0;
	for( int k=0; k<=limk; k++ )
	{
		int v = ( iin <= 0 || jin <= 0 ) ? lstop : (int)ijp[(size_t)iin*W + ((jin-1)%C)*32 + (jin-1)/C];
		if( v >= l1 + l2 ) { status = 2; break; }
		else if( v < 0 ) { ifi = iin-1; jfi = jin+v; }
		else if( v > 0 ) { ifi = iin-v; jfi = jin-1; }
		else { ifi = iin-1; jfi = jin-1; }
		int l = iin - ifi;
		while( --l > 0 ) { --pos; m1[pos] = s1[ifi+l]; m2[pos] = gap; k++; }
		l = jin - jfi;
		while( --l > 0 ) { --pos; m1[pos] = gap; m2[pos] = s2[jfi+l]; k++; }
		if( iin <= 0 || jin <= 0 ) break;
		--pos; m1[pos] = s1[ifi]; m2[pos] = s2[jfi];
		int nv = ( ifi <= 0 || jfi <= 0 ) ? lstop : (int)ijp[(size_t)ifi*W + ((jfi-1)%C)*32 + (jfi-1)/C];
		if( nv == lstop ) break;
		k++;
		iin = ifi; jin = jfi;
	}
	r.off1 = ( ifi == -1 ) ? 0 : ifi;
	r.off2 = ( jfi == -1 ) ? 0 : jfi;
	r.start = pos; r.status = status;
	res[gid] = r;
}

typedef void (*kfn_t)( const unsigned char *, const char *, const int *, const PairDesc *, Params, short *, char *, Result * );
#define K4( c ) l11k<c>, l11k<c+1>, l11k<c+2>, l11k<c+3>
static const kfn_t kernels[CMAXV+1] = { NULL, NULL, l11k<2>, l11k<3>,
	K4( 4 ), K4( 8 ), K4( 12 ), K4( 16 ), K4( 20 ), K4( 24 ), K4( 28 ), K4( 32 ),
	K4( 36 ), K4( 40 ), K4( 44 ), K4( 48 ), K4( 52 ), K4( 56 ), K4( 60 ), l11k<64> };

static int gpu_state = 0;   /* 0: not tried, 1: usable, -1: unusable */
static int verbose = 0;

static int init_gpu( void )
{
	int n = 0;
	if( gpu_state ) return( gpu_state > 0 );
	gpu_state = -1;
	if( getenv( "MAFFT_NOGPU" ) ) return( 0 );
	verbose = getenv( "L11GPU_VERBOSE" ) != NULL;
	if( cudaGetDeviceCount( &n ) != cudaSuccess || n < 1 ) { cudaGetLastError(); if( verbose ) fprintf( stderr, "l11gpu: no CUDA device\n" ); return( 0 ); }
	if( cudaSetDevice( 0 ) != cudaSuccess || cudaSetDeviceFlags( cudaDeviceScheduleBlockingSync ) != cudaSuccess || cudaFree( 0 ) != cudaSuccess ) { cudaGetLastError(); if( verbose ) fprintf( stderr, "l11gpu: no CUDA context\n" ); return( 0 ); }
	gpu_state = 1;
	return( 1 );
}

#ifndef IJPMB
#define IJPMB 384
#endif
#define IJPBUDGET ( (size_t)IJPMB << 20 )   /* bytes of traceback matrix per batch */
#define MAXBATCH 8192

typedef struct
{
	cudaStream_t st;
	cudaEvent_t ev;
	PairDesc *hpd, *dpd;
	Result *hres, *dres;
	char *hout, *dout;
	short *dijp;
	int first, n, busy;
	size_t ijpcap, outcap, pdcap;
	float ms;
	cudaEvent_t ev0;
} batch_t;

#define CK( x ) do { if( ( x ) != cudaSuccess ) { fail = 1; goto done; } } while( 0 )

extern "C" int l11gpu_align( int nseq, char **seqs, const int *lens, const int *code, int ncode, const int *mtx,
                             int pen, int ext, int thr, char gapc,
                             int npairs, const int *pi, const int *pj, l11res *out )
{
	int k, p, b, next, fail = 0, njob = 0;
	int *job = NULL;
	size_t total = 0;
	uint32_t *soff = NULL;
	unsigned char *hcodes = NULL, *dcodes = NULL;
	char *dchars = NULL;
	int *dmtx = NULL;
	batch_t bt[2];

	for( p=0; p<npairs; p++ ) out[p].status = -1;
	memset( bt, 0, sizeof( bt ) );
	if( !init_gpu() ) return( 0 );

	soff = (uint32_t *)malloc( sizeof( uint32_t ) * nseq );
	for( k=0; k<nseq; k++ ) { soff[k] = (uint32_t)total; total += lens[k] + 1; }
	hcodes = (unsigned char *)malloc( total + 16 );
	for( k=0; k<nseq; k++ )
	{
		for( int i=0; i<lens[k]; i++ ) hcodes[soff[k]+i] = (unsigned char)code[(unsigned char)seqs[k][i]];
		hcodes[soff[k]+lens[k]] = 0;
	}
	CK( cudaMalloc( &dcodes, total + 16 ) );
	CK( cudaMalloc( &dchars, total + 16 ) );
	CK( cudaMalloc( &dmtx, sizeof( int ) * ( ncode+1 ) * ( ncode+1 ) ) );
	CK( cudaMemcpy( dcodes, hcodes, total, cudaMemcpyHostToDevice ) );
	for( k=0; k<nseq; k++ ) CK( cudaMemcpy( dchars + soff[k], seqs[k], lens[k] + 1, cudaMemcpyHostToDevice ) );
	CK( cudaMemcpy( dmtx, mtx, sizeof( int ) * ( ncode+1 ) * ( ncode+1 ), cudaMemcpyHostToDevice ) );

	/* pairs the kernel can take: 64 <= l2 <= 32*CMAXV, traceback values fit in int16; ordered by C (stable) */
	job = (int *)malloc( sizeof( int ) * ( npairs + 1 ) );
	for( int c=CMIN; c<=CMAXV; c++ ) for( p=0; p<npairs; p++ )
	{
		int l1 = lens[pi[p]], l2 = lens[pj[p]];
		if( l1 < 1 || l2 < 64 || l2 > 32 * CMAXV || l1 + l2 + 1 > 32000 ) continue;
		if( ( l2 + 31 ) / 32 != c ) continue;
		if( (size_t)( l1 + 1 ) * ( 32 * c ) * 2 > IJPBUDGET ) continue;
		job[njob++] = p;
	}

	for( b=0; b<2; b++ )
	{
		CK( cudaStreamCreateWithFlags( &bt[b].st, cudaStreamNonBlocking ) );
		CK( cudaEventCreateWithFlags( &bt[b].ev, cudaEventBlockingSync | ( verbose ? 0 : cudaEventDisableTiming ) ) );
		if( verbose ) CK( cudaEventCreate( &bt[b].ev0 ) );
	}
	next = 0;
	b = 0;
	while( 1 )
	{
		batch_t *cur = bt + b, *oth = bt + ( 1 - b );
		/* collect whatever cur held, then encode the next batch into it */
		if( cur->busy )
		{
			CK( cudaEventSynchronize( cur->ev ) );
			CK( cudaGetLastError() );
			cur->busy = 0;
			if( verbose ) { float ms = 0; cudaEventElapsedTime( &ms, cur->ev0, cur->ev ); fprintf( stderr, "l11gpu: batch %d pairs, gpu %.3f s\n", cur->n, ms / 1000 ); }
			for( k=0; k<cur->n; k++ )
			{
				int q = job[cur->first+k];
				Result *r = cur->hres + k;
				PairDesc *d = cur->hpd + k;
				if( r->status == 0 )
				{
					int len = ( d->l1 + d->l2 ) - r->start;
					char *m1 = cur->hout + d->outoff + r->start;
					char *m2 = cur->hout + d->outoff + ( d->l1 + d->l2 + 1 ) + r->start;
					out[q].s1 = (char *)malloc( len + 1 ); memcpy( out[q].s1, m1, len + 1 );
					out[q].s2 = (char *)malloc( len + 1 ); memcpy( out[q].s2, m2, len + 1 );
					out[q].status = 0;
				}
				else if( r->status == 1 ) { out[q].s1 = out[q].s2 = NULL; out[q].status = 1; }
				else continue; /* leave -1 */
				out[q].maxwm = r->maxwm; out[q].off1 = r->off1; out[q].off2 = r->off2;
			}
		}
		if( next >= njob )
		{
			if( !oth->busy ) break;
			b = 1 - b;
			continue;
		}
		{
			size_t ijpsz = 0, outsz = 0;
			int n = 0, c = ( lens[pj[job[next]]] + 31 ) / 32;
			while( next + n < njob && n < MAXBATCH )
			{
				int q = job[next+n], l1 = lens[pi[q]], l2 = lens[pj[q]];
				size_t a = (size_t)( l1 + 1 ) * ( 32 * c );
				if( ( l2 + 31 ) / 32 != c ) break;
				if( n && ( ijpsz + a ) * 2 > IJPBUDGET ) break;
				ijpsz += a; outsz += 2 * (size_t)( l1 + l2 + 1 );
				n++;
			}
			if( cur->ijpcap < ijpsz * 2 ) { cudaFree( cur->dijp ); cur->dijp = NULL; CK( cudaMalloc( &cur->dijp, ijpsz * 2 ) ); cur->ijpcap = ijpsz * 2; }
			if( cur->outcap < outsz )
			{
				cudaFree( cur->dout ); cudaFreeHost( cur->hout ); cur->dout = NULL; cur->hout = NULL;
				CK( cudaMalloc( &cur->dout, outsz ) ); CK( cudaMallocHost( &cur->hout, outsz ) ); cur->outcap = outsz;
			}
			if( cur->pdcap < (size_t)n )
			{
				cudaFree( cur->dpd ); cudaFreeHost( cur->hpd ); cudaFree( cur->dres ); cudaFreeHost( cur->hres );
				cur->dpd = NULL; cur->hpd = NULL; cur->dres = NULL; cur->hres = NULL;
				CK( cudaMalloc( &cur->dpd, sizeof( PairDesc ) * n ) ); CK( cudaMallocHost( &cur->hpd, sizeof( PairDesc ) * n ) );
				CK( cudaMalloc( &cur->dres, sizeof( Result ) * n ) ); CK( cudaMallocHost( &cur->hres, sizeof( Result ) * n ) );
				cur->pdcap = n;
			}
			{
				size_t io = 0, oo = 0;
				for( k=0; k<n; k++ )
				{
					int q = job[next+k];
					PairDesc *d = cur->hpd + k;
					d->s1off = soff[pi[q]]; d->s2off = soff[pj[q]];
					d->l1 = lens[pi[q]]; d->l2 = lens[pj[q]];
					d->ijpoff = (uint32_t)io; d->outoff = (uint32_t)oo;
					d->pad0 = d->pad1 = 0;
					io += (size_t)( d->l1 + 1 ) * ( 32 * c );
					oo += 2 * (size_t)( d->l1 + d->l2 + 1 );
				}
			}
			{
				Params par = { ncode, pen, ext, thr, (unsigned char)gapc, n };
				if( verbose ) CK( cudaEventRecord( cur->ev0, cur->st ) );
				CK( cudaMemcpyAsync( cur->dpd, cur->hpd, sizeof( PairDesc ) * n, cudaMemcpyHostToDevice, cur->st ) );
				kernels[c]<<< ( n + WARPS_PER_BLOCK - 1 ) / WARPS_PER_BLOCK, 32 * WARPS_PER_BLOCK, 0, cur->st >>>( dcodes, dchars, dmtx, cur->dpd, par, cur->dijp, cur->dout, cur->dres );
				CK( cudaGetLastError() );
				CK( cudaMemcpyAsync( cur->hres, cur->dres, sizeof( Result ) * n, cudaMemcpyDeviceToHost, cur->st ) );
				CK( cudaMemcpyAsync( cur->hout, cur->dout, outsz, cudaMemcpyDeviceToHost, cur->st ) );
				CK( cudaEventRecord( cur->ev, cur->st ) );
			}
			cur->first = next; cur->n = n; cur->busy = 1;
			next += n;
		}
		b = 1 - b;
	}
done:
	if( fail )
	{
		fprintf( stderr, "l11gpu: CUDA error (%s); using the CPU\n", cudaGetErrorString( cudaGetLastError() ) );
		gpu_state = -1;
	}
	for( b=0; b<2; b++ )
	{
		if( bt[b].busy ) cudaEventSynchronize( bt[b].ev );
		if( bt[b].st ) cudaStreamDestroy( bt[b].st );
		if( bt[b].ev ) cudaEventDestroy( bt[b].ev );
		if( bt[b].ev0 ) cudaEventDestroy( bt[b].ev0 );
		cudaFree( bt[b].dijp ); cudaFree( bt[b].dout ); cudaFree( bt[b].dpd ); cudaFree( bt[b].dres );
		cudaFreeHost( bt[b].hout ); cudaFreeHost( bt[b].hpd ); cudaFreeHost( bt[b].hres );
	}
	cudaFree( dcodes ); cudaFree( dchars ); cudaFree( dmtx );
	free( hcodes ); free( soff ); free( job );
	return( !fail );
}
