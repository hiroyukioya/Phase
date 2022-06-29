function [G, W,D,AW,AB,LW,LB]=LocalityPreservingDiscriminant(x, y, t , k, mode)

%%
%<<   Locality preserving discriminant { LPD }   >> 
%            with marginal Fisher criterion
%
%
% refs : 
%        IEEE Pattren. Anal. Machine. Intell.  2005
%        Proc. Conf. Advances. Neural. Inform. Proc. Systems.  2003
%        Sugiyama,    J machine learn research,  2007
%        Hu,   Knowledge-Based sys,     2009
%        Yu,   Image and Vision comput   2006
%        He,   IEEE pattern anal machine intelli,  2005
%        Yan,  IEEE pattern anal machine intell, 2007
%
%
% x = data matrix : [dimension x observation]
% y = vector indicating class (1~class)
% k = number of  nearest neighbords
% mode=1:Marginal Fisher :    mode=2: Discriminant LPP
%

%                 < Created by  H.O.    (2009)  >

%% 
QC=unique(y);
classN=length(QC);

if length(y)~=size(x,1)
    error('     x and y must have same number of rows ....')
end
if min(y)==0
    error('     y must start from 1 ....')
end

%%  PCA dimentional reduction ------------
[x] = centernormalize(x);
origx=x;

[x] = centernormalize(x);
[v,nx,latent]=princomp(x,'econ');
per=cumsum(latent)/sum(latent);
f=find(per<=0.999);
lf=length(f);
% ii=sprintf( '/---  Number of PCA subspace = %d  ---/',lf);disp(ii);
nx=nx(:,1:f(end));
v=v(:,1:f(end));
x=nx';
%%  PLS dimensional reduction ------------

% [B, T, W,C, v, u, varE] =PLSR_NIPALS_R1(x, y, 2);
% x=T';
% varE,
%%  Locality preserving discrimination-----
[d,N]=size(x);

% if classN==1
%     disp('/*****    LPP MODE   ****/');
% elseif classN>=2
%     disp('/*****    LP < Discriminant>  MODE   ****/');
% end
        
% Compute distance -------------------
dist=pdist(x','euclidean' );
sdist=squareform(dist);

% k-nearest neighbors--------------------
[B, IX]=sort(sdist);
Bs= B(2:k+1,:); 
IXs=IX(2:k+1,:);

% Affinity matrices ----------------------
AW=zeros(N,N);
AB=zeros(N,N);

for n=1:N
      sa=find(y(IXs(:,n))==y(n));
      sadist=Bs(sa,n);
      sb=setdiff(1:k,sa);
      sadist_between=Bs(sb,n);

      [a]=heatkernelembed(sadist,t);
      [b]=heatkernelembed(sadist_between,t);
      
      % Affinity graph----
      AW(IXs(sa, n), n)=a;
      AW(n, IXs(sa, n))=a;
      
      % Penalty graph----
      AB(IXs(sb,n),n)=b;
      AB(n,IXs(sb,n))=b;
end

%%  
if mode==2 & classN~=1
    % Mean vectors....
    for n=1:classN
        f=find(y==n);
        nL=length(f);
        MV(:,n)=mean(x(:,f),2);
    end
        d=pdist(MV');
        dd=squareform(d);
        ddd=exp(-dd./t);
        E=diag(sum(ddd));
        H=E-ddd;
        F=MV*H*MV';
end

%%  Find discriminants ......
 %  Laplacians-------------------------- 
Dn=diag(sum(AW));
Df=diag(sum(AB));
LW=Dn-AW;
LB=Df-AB;

XLX_W=x*LW*x';  

if classN>=2
    if mode==1 
     XLX_B=x*LB*x';
    elseif mode==2
     XLX_B=F;
    end
elseif classN==1
    XLX_B=x*Dn*x';
end

%  Generalized eigenvalue problem ---------
[V,D]=eig(XLX_B, XLX_W);  
D=diag(D);
[i,ii]=sort(D,1,'descend');
W=v*V(:,ii);
G=(origx*W);

%%
