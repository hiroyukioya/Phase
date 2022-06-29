function [VA, LP , NY]=LocalityPreservingProjection_PD(sdist, mode, k)

%% Locality preserving projection { LPP}
% ref : IEEE Pattren. Anal. Machine. Intell.  2005
%        Proc. Conf. Advances. Neural. Inform. Proc. Systems.  2003

% sdist:  distance matrix;

% x = data matrix : [dimension x observation]
% k = number of  nearest neighbords
% mode=1: simple-minded   mode=2: Heat kernel

% H.O. 2008


[N]=size(sdist,1);

% compute distance -----------------------
% dist=pdist(x');
% sdist=squareform(dist);

% k nearest neighbors---------------------
k=10;
[B, IX]=sort(sdist);
Bs= B(2:k+1,:);
IXs=IX(2:k+1,:);

% Weight matrix: W ------------------------
temW=zeros(N,N);
W=eye(N,N);

for n=1:N
    temW(IXs(:,n),n)=Bs(:,n);
end

% simple-minded Weighting....
f=find(temW~=0);
W(f)=1;

if mode==2
    % Heat kernel  Weighting....
    m=mean(mean(temW(f)));
    t=1*m;
    W=W.*exp(-temW./t);
end


clear temW m temW

% D Matrix--------------------------------------
D=diag(sum(W));  

% Laplacian matrix L -------------------------
L=D-W;

% Generalized eigenvalue ---------------------
A=x*L*x';   A=(A+A')/2;
B=x*D*x';   B=(B+B')/2;
[eigVec , eivalue]=eig(A,B);

eivalue=diag(eivalue);
VA=eigVec;
for n=1:10
     LP(:,n)=VA(:,n)'*x;
end
NY=VA'*x;



    

