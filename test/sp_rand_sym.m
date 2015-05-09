function time = sp_rand_sym(n, density, k)

M = sprandsym(n, density);
M = spones(M);

opts.p = 20;
tic;
O = eigs(C, k, 'lm', opts);
time = toc
