// Each lane owns one lens, radial bin, and contiguous source segment.
// Private FP64 accumulation and ordered segment reduction avoid atomic sums.
extern "C" __global__ void pairs(
 const double* l, const double* s, const double* edges, const double* eff,
 const long long* zb, const long long* tasks, long long nl, long long ns,
 long long ntask, int nb, int nz, int comoving,
 double weighting, double factor, double* out, long long* counts) {
 long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(tid>=ntask*nb) return;
 int b=tid%nb;
 long long task=tid/nb, il=tasks[3*task];
 double a[11]={0}; long long count=0;
 #define L(k) l[(k)*nl+il]
 #define S(k) s[(k)*ns+is]
 for(long long is=tasks[3*task+1];is<tasks[3*task+2];++is) {
  if(L(0)>S(9)) continue;
  double dx=L(3)*L(5)-S(3)*S(5), dy=L(2)*L(5)-S(2)*S(5), dz=L(4)-S(4);
  double d=dx*dx+dy*dy+dz*dz;
  if(!(d>edges[il*(nb+1)+b] && d<=edges[il*(nb+1)+b+1]) || S(6)==0) continue;
  double sc;
  if(nz) sc=eff[il*nz+zb[is]];
  else if(L(1)<S(1)) {
   sc=factor*(1+L(0))*S(1)/L(1)/(S(1)-L(1));
   if(comoving) sc/=(1+L(0))*(1+L(0));
  } else { if(weighting<0) continue; sc=__longlong_as_double(0x7ff0000000000000LL); }
  double w=weighting==0 ? S(6) : (weighting==-2 ? S(6)/sc/sc : S(6)*pow(sc,weighting));
  if(w==0) continue;
  double sr=L(2)*S(3)-L(3)*S(2), cr=L(3)*S(3)+L(2)*S(2);
  double num=S(5)*L(4)-S(4)*L(5)*cr, den=L(5)*sr;
  double co=-1, si=0;
  if(den!=0) { double t=num/den; co=2/(1+t*t)-1; si=2*t/(1+t*t); }
  double et=-S(7)*co+S(8)*si, ex=-S(8)*co-S(7)*si;
  ++count;
  a[0]+=w; a[1]+=w*et; a[2]+=w*S(0); a[3]+=(w*et)*sc; a[4]+=w*sc;
  a[5]+=w*S(10); a[6]+=w*(1-S(11)*S(11)); a[7]+=w*S(12);
  a[8]+=w*(S(13)*co*co+S(16)*si*si+(S(14)+S(15))*si*co);
  a[9]+=w*ex; a[10]+=(w*ex)*sc;
 }
 counts[tid]=count;
 for(int k=0;k<11;++k) out[(tid/nb*11+k)*nb+b]=a[k];
}

// Explicit segment loops avoid prefix-sum subtraction (which would propagate
// infinities from one lens to another) and fix the reduction order.
extern "C" __global__ void segmented_double(const double* values,
 const long long* starts, long long nrow, long long nseg, long long width,
 double* output) {
 long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(tid>=nseg*width) return;
 long long group=tid/width, col=tid%width;
 long long end=group+1<nseg ? starts[group+1] : nrow;
 double total=0;
 for(long long row=starts[group];row<end;++row) total+=values[row*width+col];
 output[tid]=total;
}
extern "C" __global__ void segmented_int64(const long long* values,
 const long long* starts, long long nrow, long long nseg, long long width,
 long long* output) {
 long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(tid>=nseg*width) return;
 long long group=tid/width, col=tid%width;
 long long end=group+1<nseg ? starts[group+1] : nrow;
 long long total=0;
 for(long long row=starts[group];row<end;++row) total+=values[row*width+col];
 output[tid]=total;
}

extern "C" __global__ void grouped_finite(const double* values,
 const long long* starts, long long nrow, long long nseg, long long width,
 double* finite, long long* flags) {
 long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(tid>=nseg*width) return;
 long long group=tid/width, col=tid%width;
 long long end=group+1<nseg ? starts[group+1] : nrow;
 double sum=0; long long nan=0, pos=0, neg=0, nonzero=0;
 for(long long row=starts[group];row<end;++row) {
  double v=values[row*width+col];
  if(isfinite(v)) sum+=v;
  else if(isnan(v)) ++nan;
  else if(v>0) ++pos;
  else ++neg;
  if(v!=0) ++nonzero;
 }
 finite[tid]=sum;
 flags[tid]=nan;
 flags[nseg*width+tid]=pos;
 flags[2*nseg*width+tid]=neg;
 flags[3*nseg*width+tid]=nonzero;
}
