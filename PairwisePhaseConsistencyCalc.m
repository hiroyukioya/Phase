function [Y]=PairwisePhaseConsistencyCalc(theta)
% Pairwise phase condistency
% theta1 is a vector with complex number represeting phase
% H.O. 2010
% Rev. Christopher Kovach


[a,b]=size(theta);
tabs=abs(theta);
theta=theta./tabs; % make theta unit vector

% new version for speeding up
R = repmat(theta,[1 1 a]).*repmat(permute(1-eye(a),[1 3 2]),1,size(theta,2));
D = R.*conj(permute(R,[ 3 2 1]));
Y = real(sum(sum(D,3)))/(a*(a-1)/2)/2;

% Original version
% for m=1:b
%     Hr=hankel(real(theta(:,m)'));    
%     Hi=hankel(imag(theta(:,m)'));
%     H1=Hr(1,:)*Hr;
%     H2=Hi(1,:)*Hi;
%     s=[H1(1,2:end)  H2(1,2:end)];
%     Y(m)=sum(s)/(a*(a-1)/2);
% end
