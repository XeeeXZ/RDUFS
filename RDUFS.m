function [w,H,out] = RDUFS(X,XtLX,opts)
% RDUFS: Solve diagonal sparse feature selection model
%   max 0.5*MCC(X,Theta,H) - gamma/2 *Tr(Theta'*XtLX*Theta) - lambda*||Theta||_1
%   subject to Theta is diagonal, Theta >= 0
%
% Input:
%   X:      Input data matrix (n × d)
%   XtLX:   Precomputed X'*L*X, where L is Laplacian
%   opts:   Structure with parameters:
%           - tol: tolerance for convergence (default 1e-7)
%           - maxit: max iterations (default 50)
%           - Theta0: initial Theta (default eye(d))
%           - H0: initial H (default random)
%           - gamma: Laplacian regularization weight
%           - lambda: L1 sparsity weight
%           - theta: rho scaling factor
% Notation:
% For notational convenience, W in this code denotes the diagonal
% feature-weighting matrix Theta in the manuscript.
% The vector w contains its diagonal entries, i.e., W = diag(w).

%% Parameters and defaults
if isfield(opts,'tol'),    tol = opts.tol;     else tol = 1e-7;   end
if isfield(opts,'maxit'),  maxit = opts.maxit; else maxit = 50;  end
if isfield(opts,'rw'),     rw = opts.rw;       else rw = 1;       end
if isfield(opts,'gamma'),  gamma = opts.gamma;       else gamma = 1;       end
if isfield(opts,'lambda'),    lambda = opts.lambda;   else lambda = 0.5;   end
if isfield(opts,'theta'),   theta = opts.theta;   else theta = 1;   end

%% Data preprocessing and initialization
[nSmp, nFea] = size(X);
w0 = opts.w0;

%  W0 = opts.W0;

H0 = opts.H0;

wm = w0; H = H0;
Ht = H0'; Hs = H0*Ht;


% Xs = X'*X;
XW = X.*(w0');
ErrorM = X-XW*H0;
Error = sum(ErrorM.^2,2);
% sigma2 = 5;
sigma2 = (theta/(2*nSmp))*sum(Error);
rho = exp((-Error)/sigma2);
% rho = exp((-Error)/theta);
% nrmX = norm(X,'fro');
% if isfield(opts, 'normXs'), normXs = opts.normXs; else normXs = norm(Xs); end
if isfield(opts, 'normXtLX'), normXtLX = opts.normXtLX; else normXtLX = gamma*norm(XtLX); end
ErrorMm = ErrorM.*repmat(sqrt(rho),1,nFea);
obj0 = 0.5*sum(ErrorMm(:).^2);
obj2 = (gamma/2) * sum(diag(XtLX) .* (w0.^2));
obj3 = lambda * sum(abs(w0));  
obj0 = obj0+obj2+obj3;

obj_totle0 = 0.5*sum(rho)-obj2-obj3;

t0 = 1;
LW = 1;
hist_LW = [];
hist_rho = [];

%% Iteration of block-coordinate upadate

for k = 1:maxit
    %--- W-update ---
    LW0 = LW;
    Xt = X';
    Xs = Xt * (X.*repmat(rho,1,nFea));
    norm_Xs = norm(Xs, 2);      
    norm_HHt = norm(Hs, 2);    
    
    LW = norm_Xs * norm_HHt + normXtLX;
    hist_LW = [hist_LW; LW];

   
    term1 = sum(Xs .* (Hs.' .* (wm.')), 2);   

    term2 = sum(H.' .* Xs, 1).';             

    term3 = gamma * diag(XtLX) .* wm;       

    grad = term1 - term2 + term3;

    w      = wm - grad / LW;               
    w      = max(0, w - lambda / LW);     

    XsWt = Xs .* w;
    XWs  = Xs .* (w * w.');

    %--- H-update ---
    %     LH0 = LH;
    epsilon = 1e-6;
    H = (XWs + epsilon * eye(size(XWs))) \ XsWt;
    Ht = H'; Hs = H*Ht;

    %--- Error-update ---
    XW    = X .* (w.');
    ErrorM = X-XW*H;
    ErrorMm = ErrorM.*repmat(sqrt(rho),1,nFea);

    %--- Check whether the update is right ---
    obj1 = 0.5*sum(ErrorMm(:).^2);
    obj2  = (gamma/2) * sum( diag(XtLX) .* (w.^2) );  
    obj3  = lambda * sum(abs(w));                     
    obj = obj1+obj2+obj3;

    t = (1+sqrt(1+4*t0^2))/2;
    if obj>=obj0
        % restore to previous W,H, and cached quantities for nonincreasing
        % objective

        %--- W-update ---
        wm = w0;H = H0;
        Ht = H0'; Hs = H0*Ht;
        LW0 = LW;
        Xt = X';
        Xs = Xt * (X.*repmat(rho,1,nFea));
        norm_Xs = norm(Xs, 2);       
        norm_HHt = norm(Hs, 2);    
        
        LW = norm_Xs * norm_HHt + normXtLX;

        hist_LW = [hist_LW; LW];

        term1 = sum(Xs .* (Hs.' .* (wm.')), 2);  

        term2 = sum(H.' .* Xs, 1).';             

        term3 = gamma * diag(XtLX) .* wm;        

        grad = term1 - term2 + term3;

        w      = wm - grad / LW;               
        w      = max(0, w - lambda / LW);     

        XsWt = Xs .* w;
        XWs  = Xs .* (w * w.');

        %--- H-update ---
        %     LH0 = LH;
        epsilon = 1e-6;
        H = (XWs + epsilon * eye(size(XWs))) \ XsWt;
        Ht = H'; Hs = H*Ht;

        %--- Error-update ---
        XW    = X .* (w.');
        ErrorM = X-XW*H;
        ErrorMm = ErrorM.*repmat(sqrt(rho),1,nFea);

        obj1 = 0.5*sum(ErrorMm(:).^2);
        obj2  = (gamma/2) * sum( diag(XtLX) .* (w.^2) );   
        obj3  = lambda * sum(abs(w));                    
        obj = obj1+obj2+obj3;


        % extrapolation
        omega  = (t0 - 1)/t;
        omegaW = min([omega, rw*sqrt(LW0/LW)]);
        wm     = w + omegaW * (w - w0);
        w0 = w; H0 = H; t0 = t; obj0 = obj;

    else
        % extrapolation
        omega  = (t0 - 1)/t;
        omegaW = min([omega, rw*sqrt(LW0/LW)]);
        wm     = w + omegaW * (w - w0);
        w0 = w; H0 = H; t0 = t; obj0 = obj;
    end

    %--- rho-update ---
    Error = sum(ErrorM.^2,2);
    %     sigma2 = 5;
    sigma2 = (theta/(2*nSmp))*sum(Error);
    rho = exp((-Error)/sigma2);
    %     rho = exp((-Error)/theta);

    %    hist_rho = [hist_rho, rho];

    %--- diagnostics, reporting, stopping checks ---
    obj_totle = 0.5*sum(rho);
    obj_totle = obj_totle-obj2-obj3;

    %reporting
    out.hist_obj(k) = obj_totle;
    out.relerr(k) = abs(obj_totle-obj_totle0)/(obj_totle0+1);
    obj_totle0 = obj_totle;

    % stall and stoping checks
    %     crit = (out.relerr(k)<tol);
    %     if crit; nstall = nstall+1; else nstall = 0; end
    %     if nstall >=3;  break; end

end
out.iter = k;
out.hist_LW = hist_LW;
out.hist_rho = hist_rho;


end