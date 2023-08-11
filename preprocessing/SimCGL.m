% ********************************************************************
% Complex Ginzburg-Landau with Gaussian noise and periodic
% boundary conditions. Simulated using pseudospectral method 
% and ETD2 exponential time-stepping. Adapted by C.-W. Joshua Liu
% from D.M. Winterbottom. The complex Ginzburg-Landau equation (2005)
% ********************************************************************

% Set iterator parameters
for c1  = -0.2                        % Linear diffusivity
for c2  = 0.5                         % Nonlinear diffusivity
for s   = 0                           % Seed
for dT  = 0.1

    disp('*** CGL PHASE FIELD SIMULATION ***');
    disp(['c1 = ' num2str(c1) ', c2 = ' num2str(c2)...
          ', s = ' num2str(s) ', dT = ' num2str(dT)])

    % Set system parameters
    sigma   = 0;         % Gaussian noise sigma 
    L       = 100;       % Domain size (assume square container)
    Tmax    = 100000;    % End time
    N       = 64;        % Number of grid points
    dps     = 10000;     % Number of stored times
    
    % Calculate derived parameters
    nmax  = round(Tmax/dT);
    XX      = (L/N)*(-N/2:N/2-1); 
    [X,Y] = meshgrid(XX);
    
    % Define initial conditions
    rng(s)
    Tdata = zeros(1,dps);
    Tdata(1) = 0;
    A = zeros(size(X)) + 10^(-4)*randn(size(X));
    
    % Set wavenumbers
    k  				 = [0:N/2-1 0 -N/2+1:-1]*(2*pi/L);
    k2 				 = k.*k;
    k2(N/2+1)        = ((N/2)*(2*pi/L))^2;
    [k2x,k2y]        = meshgrid(k2);
    del2 			 = k2x+k2y;
    Adata            = zeros(N,N,dps);
    A_hatdata        = zeros(N,N,dps);
    A_hat            = fft2(A);
    Adata(:,:,1)     = A;
    A_hatdata(:,:,1) = A_hat;
    
    % Compute exponentials and nonlinear factors for ETD2 method
    cA 	    	= 1 - del2*(1+1i*c1);
    expA 	  	= exp(dT*cA);
    nlfacA  	= (exp(dT*cA).*(1+1./cA/dT)-1./cA/dT-2)./cA;
    nlfacAp 	= (exp(dT*cA).*(-1./cA/dT)+1./cA/dT+1)./cA;
    
    % Solve PDE
    dataindex = 1;
    for n = 1:nmax
	    T = Tdata(1) + n*dT;
	    A = ifft2(A_hat)+sigma.*(randn(N)+1i.*randn(N));
	    
	    % Find nonlinear component in Fourier space
	    nlA	= -(1+1i*c2)*fft2(A.*abs(A).^2);
	    
	    % Setting the first values of the previous nonlinear coefficients
	    if n == 1
		    nlAp = nlA;
	    end
	    
	    % Time-stepping
	    A_hat = A_hat.*expA + nlfacA.*nlA + nlfacAp.*nlAp;
	    nlAp  = nlA;
	    
	    % Save data
        if nmax-n < dps
		    A = ifft2(A_hat);
		    Adata(:,:,dataindex)     = A; 
		    A_hatdata(:,:,dataindex) = A_hat; 
		    Tdata(dataindex)         = T;
		    dataindex                = dataindex + 1;
        end
	    
        % Display steps completed
	    if mod(n, floor(nmax/10)) == 0
		    disp(strcat(num2str(n), ' steps completed'));
	    end
    end
    
    % Save file
    disp('Writing file...')
    phi = permute(angle(Adata), [3 1 2]);
    im = uint8((phi+pi).*255/(2*pi));
    for j = 1:dps
        imwrite(reshape(im(j, :, :), [64 64]),...
                strcat(erase(sprintf('c1_%.2f_c2_%.2f_dT_%.2f_s_%d',...
                c1, c2, dT, s), '.'), '.tif'), 'WriteMode', 'append');
    end
    disp('File written')
end
end
end
end