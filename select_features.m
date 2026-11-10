function [Ind, history] = select_features(X, K, XtLX, gamma, lambda, options)

[nSmp, nFea] = size(X);
Ind = cell(1, length(K));

opts.gamma = gamma;
opts.lambda = lambda;
opts.maxit = 100;
opts.tol   = 1e-6;

if isfield(options, 'w0'), opts.w0 = options.w0; end
if isfield(options, 'H0'), opts.H0 = options.H0; end
if isfield(options, 'theta'), opts.theta = options.theta; end

[w, H, out] = RDUFS(X, XtLX, opts);

history.w = w;
history.H = H;
history.out = out;

% figure;
% plot(history.out.hist_obj, '-o', 'LineWidth', 1.5);
% xlabel('Iteration');
% ylabel('Objective Function Value');
% title(sprintf('MFSMCC Convergence Curve (\\theta = %.2f)', opts.theta));
% grid on;

[Value, I] = sort(w, 'descend');

for ii = 1:length(K)
    if Value(K(ii)) == 0
        Ind{ii} = I(Value ~= 0);  
    else
        Ind{ii} = I(1:K(ii));
    end
    fprintf('Select top-%3d features done.\n', K(ii));
end

end
