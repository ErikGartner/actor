% stereoReconsPts - reconstruct a set of 3D points from their 2-view image projections and the corresponding camera matrices
%
%    pts3D = stereoReconsPts(P0, P1, pts0, pts1, imgsizes, howto, exarg)
% 
%    P0, P1     - 3x4 camera matrices (finite)
%    pts0, pts1 - 2xn corresponding image projections
%    imgsizes   - (optional) 2x2 matrix whose columns are image sizes, used for preconditioning in DLT;
%                 e.g. [640 640; 480 480] for 640x480 images or [] if no preconditioning is desired.
%                 A value of [-1] performs an isotropic normalization for each of pts0, pts1 as suggested by Hartley
%                 A value of [-2] performs a simple scaling to the DLT matrix
%    howto      - optional argument that specifies how projections are to be corrected prior to triangulation:
%                    * if 'poly', projections are optimally corrected by minimizing the reprojection error, thus
%                          they comply with the epipolar geometry (solving a 6th deg. polynomial);
%                    * if 'sampson', projections are approximately corrected by minimizing the Sampson approximation
%                          of the reprojection error; the epipolar geometry is not satisfied exactly
%                    * if 'lind', projections are corrected by Lindstrom's fast method
%                    * if 'midp', no correction is attempted and points are reconstructed with the midpoint algorithm
%                    * if 'none' no correction is attempted and points are reconstructed with DLT
%                    * no correction is performed otherwise (default)
%    exarg      - extra argument that depends on the value of 'howto': for 'poly' & 'sampson' it should be the fundamental matrix;
%                 computed from the camera matrices if not supplied
%
% Returns
%    pts3D      - 3xn reconstructed 3D points (Euclidean)

% see HZ2, p.312, 314, 318 & "Triangulation" by HS
%
% Manolis Lourakis 2007-18
% Institute of Computer Science, Foundation for Research & Technology - Hellas
% Heraklion, Crete, Greece

% April 2018  - Original version. (v. 1.0)
% July  2018  - Added midpoint triangulation. (v. 1.1)
% Nov   2018  - Added Lindstrom's correction. (v. 1.2)

function [pts3D] = stereoReconsPts(P0, P1, pts0, pts1, imgsizes, howto, exarg)

  if((size(pts0, 1)~=2 && size(pts0, 1)~=3) || (size(pts1, 1)~=2 && size(pts1, 1)~=3))
    error('stereoReconsPts: can only handle 2D correspondences');
  end

  if(size(pts0, 2)~=size(pts1, 2))
    error('stereoReconsPts: different number of 2D point pairs in the two sets');
  end

  if(nargin<=4)
    imgsizes=[];
  else
    if(~isnumeric(imgsizes))
      error('stereoReconsPts: "imgsizes" must be a numeric array!');
    end
  end

  if(nargin<=5)
    howto='none';
  else
    if(~ischar(howto))
      error('stereoReconsPts: "howto" must be a string!');
    end
    howto=lower(howto);
  end

  if(size(pts0, 1)==3)
    % make point sets inhomogeneous: divide by third coord if non-unit
    if(sum(pts0(3,:)~=1)>0), pts0=pts0(1:2,:)./repmat(pts0(3,:), 2,1); else, pts0=pts0(1:2,:); end
    if(sum(pts1(3,:)~=1)>0), pts1=pts1(1:2,:)./repmat(pts1(3,:), 2,1); else, pts1=pts1(1:2,:); end
  end

  if(strncmp(howto, 'poly', 3) || strncmp(howto, 'sampson', 3))
    if(nargin>=7)
      F=exarg;
    else
      warning('stereoReconsPts: F not supplied, extracting from camera matrices');

      % HZ2 p.246
      % homography converting P0, P1 to the canonical form
      M=P0(1:3,1:3); b=P0(:,4);
      if(cond(M)>1E15)
        error('stereoReconsPts: first camera must be finite!');
      end
      %M=inv(M); H=[M -M*b; 0 0 0 1]; PP1=P1*H;
      PP1=P1/[M b; 0 0 0 1];
      M=PP1(1:3,1:3); b=PP1(1:3,4);
      F=[0, -b(3), b(2); b(3), 0, -b(1); -b(2), b(1), 0]*M;
    end
  end

  if(strncmp(howto, 'poly', 3))
    [cpts0, cpts1]=poly6cor(F, pts0, pts1);
    pts3D=lintriang(P0, P1, cpts0, cpts1, imgsizes);
  elseif(strncmp(howto, 'sampson', 3))
    [cpts0, cpts1]=sampsoncor(F, pts0, pts1);
    pts3D=lintriang(P0, P1, cpts0, cpts1, imgsizes);
  elseif(strncmp(howto, 'lind', 4))
    % extract calibration and essential matrices from Ps
    [K0 R0]=myrq(P0(1:3,1:3)); invK0=inv(K0); t0=invK0*P0(:,4);
    [K1 R1]=myrq(P1(1:3,1:3)); invK1=inv(K1); t1=invK1*P1(:,4);

    R=R1*R0'; t=t1-R*t0;
    E=[0, -t(3), t(2); t(3), 0, -t(1); -t(2), t(1), 0]*R;

    % normalize (3xn)
    npts=size(pts0, 2);
    npts0=invK0*[pts0; ones(1, npts)];
    npts1=invK1*[pts1; ones(1, npts)];

    [cnpts0, cnpts1]=lindstromcor(E, npts0, npts1);

    % convert back to pixels (2xn)
    cpts0=K0*cnpts0; cpts0=cpts0(1:2,:)./repmat(cpts0(3,:), 2,1);
    cpts1=K1*cnpts1; cpts1=cpts1(1:2,:)./repmat(cpts1(3,:), 2,1);
    pts3D=lintriang(P0, P1, cpts0, cpts1, imgsizes);
  elseif(strncmp(howto, 'midp', 3))
    pts3D=midptriang(P0, P1, pts0, pts1);
  elseif(strncmp(howto, 'none', 3))
    pts3D=lintriang(P0, P1, pts0, pts1, imgsizes);
  else
    error('stereoReconsPts: unknown projections correction method!');
  end
end

% optimally correct points so that they satisfy the epipolar constraint (a la Hartley & Sturm)
function [corpts0, corpts1] = poly6cor(F, pts0, pts1)

  npts=size(pts0, 2);

  T=eye(3, 3); Tp=eye(3, 3);
  R=eye(3, 3); Rp=eye(3, 3);
  T1=eye(3, 3); T1p=eye(3, 3);
  corpts0=zeros(2, npts);
  corpts1=zeros(2, npts);
  cf=zeros(7,1);
  xc=zeros(3,1);
  for i=1:npts
    T (1:2,3)=-pts0(:,i);
    Tp(1:2,3)=-pts1(:,i);
    T1 (1:2,3)=pts0(:,i); % T1 =inv(T);
    T1p(1:2,3)=pts1(:,i); % T1p=inv(Tp);
    Fp=T1p'*F*T1;
    [U,dummy,V]=svd(Fp,0);
    e=V(:,3);  e=  e./norm(e(1:2));
    ep=U(:,3); ep=ep./norm(ep(1:2));

    R(1:2,1:2)= [e(1:2)';  -e(2) e(1)];
    Rp(1:2,1:2)=[ep(1:2)'; -ep(2) ep(1)];
    Fp=Rp*Fp*R';

    f=e(3); fp=ep(3); a=Fp(2,2); b=Fp(2,3); c=Fp(3,2); d=Fp(3,3);

    % polynomial 12.7
    t1=a*d; t2=b*c; t3=t1-t2; t4=f*f;
    t5=t4*t4; t6=t3*t5; t7=a*c;
    cf(1)=-t6*t7;
    t9=a*a; t10=fp*fp; t11=c*c;
    t13=t9+t10*t11; t14=t13*t13;
    cf(2)=t14-t6*t2-t6*t1;
    t21=2.0*(b*a+t10*d*c); t24=t3*t4; t27=b*d;
    cf(3)=2.0*(t21*t13-t24*t7)-t6*t27;
    t29=b*b; t30=d*d; t32=t29+t10*t30; t35=t21*t21;
    cf(4)=2.0*(t32*t13-t24*t2-t24*t1)+t35;
    t42=t3*a;
    cf(5)=2.0*(t32*t21-t24*t27)-t42*c;
    t46=t32*t32; t47=t3*b;
    cf(6)=t46-t47*c-t42*d;
    cf(7)=-t47*d;

    r=roots(cf);
    r=r(imag(r)==0); % discard non-real

    s=arrayfun(@(t) (t*t)/(1.0+(f*t)^2) + (c*t+d)^2/((a*t+b)^2 + (fp*(c*t+d))^2), r); % 12.5
    % should also check the asymptotic value at \infty: 1/(f*f) + c*c/(a*a + (fp*c)^2);
    tmin=min(s);

    lam=tmin*f; nu=-tmin; % mu==1
    xc=[-lam*nu, -nu, lam*lam+1]';
    xc=T1*R'*xc;
    corpts0(:,i)=xc(1:2)./xc(3);

    lam=-fp*(c*tmin+d); mu=a*tmin+b; nu=c*tmin+d;
    xc=[-lam*nu, -mu*nu, lam*lam+mu*mu]';
    xc=T1p*Rp'*xc;
    corpts1(:,i)=xc(1:2)./xc(3);
  end

end

% correct points using the Sampson approximation; the epipolar constraint is not satisfied exactly!
function [corpts0, corpts1] = sampsoncor(F, pts0, pts1)

  Fn=F./norm(F(:));

  npts=size(pts0, 2);
  for i=1:npts
    r=[pts1(:,i)' 1]*Fn*[pts0(:,i); 1];

    %fd0=Fn(1:3,1:2)'*[pts1(:,i); 1];
    fd0=Fn(1:2,1:2)'*[pts1(:,i)] + Fn(3,1:2)';

    %fd1=Fn(1:2,1:3)*[pts0(:,i); 1];
    fd1=Fn(1:2,1:2)*[pts0(:,i)] + Fn(1:2,3);

    g=fd0'*fd0 + fd1'*fd1;
    e=r/g;

    corpts0(:,i)=pts0(:,i) - e*fd0;
    corpts1(:,i)=pts1(:,i) - e*fd1;
  end

end


% linear two-view triangulation see HZ2, p. 312
% imgsizes are the (optional) image dimensions, used for preconditioning, e.g. [512 512; 384 384] for 512x384 images;
% a 2-vector indicates identical dimensions for both images;
% a value of -1 indicates normalization as suggested by Hartley
% a value of -2 indicates scaling each column of the design matrix with its largest element
function [pts3D] = lintriang(P0, P1, pts0, pts1, imgsizes)

  %{
  if(size(pts0, 1)~=2 || size(pts1, 1)~=2)
    error('lintriang: can only handle 2D correspondences');
  end

  if(size(pts0, 2)~=size(pts1, 2))
    error('lintriang: different number of 2D point pairs in the two sets');
  end
  %}

  npts=size(pts0, 2);

  A=zeros(2*2, 4);
  pts3D=zeros(3, npts);

  if(nargin>4 && ~isempty(imgsizes))
    if(prod(imgsizes(:)==-1)) % normalize a la Hartley
      H0=normalize2D(pts0);
      H1=normalize2D(pts1);
    elseif(prod(imgsizes(:)==-2)) % use a quick scaling for A; rest of code same as below
      for i=1:npts
        A=[pts0(:,i)*P0(3,:)-P0(1:2,:); pts1(:,i)*P1(3,:)-P1(1:2,:)];
        D=diag(1./max(abs(A))); % diagonal scaling matrix: A = A*D*inv(D)
        A=A*D;
        [dummy dummy V]=svd(A, 0);
        V(:,4)=D*V(:,4); % undo normalization
        pts3D(:,i)=V(1:3, 4)./V(4, 4); % the conditioning of the estimated point is S(3,3)/S(4,4), S being the singular values matrix
      end
      return;
    else % normalize a la VGG
      [r c]=size(imgsizes);
      if(r~=2 || c~=2)
        if(r*c==2)
          %warning('lintriang: assuming identical sizes for both images');
          imgsizes=[imgsizes(:) imgsizes(:)];
        else
          error('lintriang: "imgsizes" must be a 2x2 array');
        end
      end
      H0=[2/imgsizes(1, 1) 0 -1
         0 2/imgsizes(2, 1) -1
         0 0 1];
      H1=[2/imgsizes(1, 2) 0 -1
         0 2/imgsizes(2, 2) -1
         0 0 1];
    end

    % left multiply Ps and points
    P0=H0*P0;
    P1=H1*P1;
    pts0=H0(1:2, 1:2)*pts0 + repmat(H0(1:2, 3), 1, npts);
    pts1=H1(1:2, 1:2)*pts1 + repmat(H1(1:2, 3), 1, npts);
  end

  % at this point, normalization has been performed with the H's above
  for i=1:npts
    A=[pts0(:,i)*P0(3,:)-P0(1:2,:); pts1(:,i)*P1(3,:)-P1(1:2,:)];

    % Linear-Eigen
    [dummy dummy V]=svd(A, 0);
    pts3D(:,i)=V(1:3, 4)./V(4, 4);

    % this should be a bit faster but not as stable
    %[V dummy]=eig(A'*A);
    %pts3D(:,i)=V(1:3, 1)./V(4, 1);

    % Linear-LS (4th element assumed 1)
    %pts3D(:,i)=-A(:,1:3)\A(:, 4);

  end

end


% two-view triangulation with the mid-point method. Determines the closest 3D point between the
% two rays. Fast but suboptimal with respect to the reprojection error; see Hartley & Sturm
function [pts3D] = midptriang(P0, P1, pts0, pts1)

  %{
  if(size(pts0, 1)~=2 || size(pts1, 1)~=2)
    error('midptriang: can only handle 2D correspondences');
  end

  if(size(pts0, 2)~=size(pts1, 2))
    error('midptriang: different number of 2D point pairs in the two sets');
  end
  %}

  npts=size(pts0, 2);

  A=zeros(3, 2);
  b=zeros(3, 1);
  pts3D=zeros(3, npts);

  M0=P0(1:3, 1:3);
  M0=inv(M0);
  c0=-M0*P0(:, 4);

  M1=P1(1:3, 1:3);
  M1=inv(M1);
  c1=-M1*P1(:, 4);

  for i=1:npts
    A=[M0*[pts0(:,i);1], -M1*[pts1(:,i);1]];
    b=c1-c0;

    [Q, R]=qr(A); a=R\(Q'*b); % LS with QR
    pts3D(:,i)=(c0+a(1)*A(:,1) + c1-a(2)*A(:,2))*0.5;

    % alternative solution; see Ramalingam, Lodha & Sturm
    %{
    A=zeros(3,3);
    u=M0*[pts0(:,i);1];
    A=eye(3,3) - (u*u')./(u'*u);
    b=A*c0;

    u=M1*[pts1(:,i);1];
    tmp=eye(3,3) - (u*u')./(u'*u);
    A=A+tmp;
    b=b+tmp*c1;
    pts3D(:,i)=A\b;

    %[U,S,V]=svd(A);
    %c=U'*b;
    %pts3D(:,i)=V*[c(1)/S(1,1); c(2)/S(2,2); c(3)/S(3,3)];
    %}
  end

end


% compute the transformation translating and scaling a set of 2D points so that
% their centroid is at the origin and their mean distance from it is sqrt(2)
function T = normalize2D(pts)

  %{
  if(size(pts,1)~=2)
    error('normalize2D: can only handle 2D correspondences');
  end
  %}

  npts=size(pts,2);
    
  % compute centroid and shift origin
  %m=mean(pts')';
  m=mean(pts, 2);
  nrmpts=pts - repmat(m, 1, npts);
    
  dist=sqrt(nrmpts(1, :).^2 + nrmpts(2, :).^2);
  meandist=mean(dist(:));
    
  scale=sqrt(2)/meandist;
    
  T=[scale   0   -scale*m(1)
     0     scale -scale*m(2)
     0       0      1      ];

  %nrmpts=scale*pts + repmat(T(1:2, 3), 1, npts);
end


% lindstromcor - correct points with Lindstrom's fast method
%
%    [cornpts0, cornpts1] = lindstromcor(E, npts0, npts1)
% 
%    E            - 3x3 essential matrix
%    npts0, npts1 - 2xn corresponding normalized image projections

% see Lindstrom: "Triangulation Made Easy"

function [cornpts0, cornpts1] = lindstromcor(E, npts0, npts1)

  %{
  if(size(npts0, 1)~=2 || size(npts1, 1)~=2)
    error('lindstromcor: can only handle 2D correspondences');
  end

  if(size(npts0, 2)~=size(npts1, 2))
    error('lindstromcor: different number of 2D point pairs in the two sets');
  end
  %}

  S=[1 0 0; 0 1 0];
  Etilde=S*E*S';
  npts=size(npts0, 2);
  cornpts0=zeros(3, npts);
  cornpts1=zeros(3, npts);
  for i=1:npts
    x =npts0(:,i);
    xp=npts1(:,i);

    n =S*E*xp;
    np=S*E'*x;
    a=n'*Etilde*np;
    b=0.5*(n'*n + np'*np);
    c=x'*E*xp;
    d=sqrt(b*b-a*c);
    lam=c/(b+d);
    dx =lam*n;
    dxp=lam*np;
    n =n  - Etilde*dxp;
    np=np - Etilde'*dx;

    % niter1
    %{
    dx= (dx'*n)./(n'*n)*n;
    dxp=(dxp'*np)./(np'*np)*np;
    %}

    % niter2
    %lam=lam*2*d/(n'*n + np'*np);
    lam=lam*d/b; % the denom. above is twice b; 2's cancel out
    dx =lam*n;
    dxp=lam*np;

    cornpts0(:,i)=x  - S'*dx;
    cornpts1(:,i)=xp - S'*dxp;
  end

end


% RQ decomposition of 3x3 matrix
% see https://math.stackexchange.com/questions/1640695/rq-decomposition
% Kovesi's rq3 should be faster
function [R,Q]=myrq(A)
  P=[0 0 1; 0 1 0; 1 0 0];
  [Q,R]=qr((P*A)');
  R=P*R'*P;
  Q=P*Q';

  % make sure that the diagonal elements of R are positive
  for n=1:3
    if R(n,n)<0
      R(:,n)=-R(:,n);
      Q(n,:)=-Q(n,:);
    end
  end

end

