%Part 1
K = 9;
w = 3;
zetas = linspace(0.1,2,20);

for i = 1:1:20
    sys = tf(K,[1 2*zetas(i)*w w^2]);
    subplot(2,1,1);
    [steps, tout] = step(sys);
    plot(tout, steps);
    title(sprintf("zeta = %f", zetas(i)));
    ylabel('h(t)')
    xlabel('t')
   
    subplot(2,1,2)
    root = roots([1 2*zetas(i)*w w^2]);
    plot(real(root), imag(root), 'o')
    title("poles")
    ylabel("Imaginary Part")
    xlabel("Real Part")
    hold on
    shg
    pause(0.1);
end
hold off

zeta = 0.5;
omegas = linspace(0.5, 20, 40);
for i = 1:1:40
    sys = tf(omegas(i)^2,[1 2*zeta*omegas(i) omegas(i)^2]);
    subplot(2,1,1);
    [steps, tout] = step(sys);
    plot(tout, steps);
    title(sprintf("omega = %f", omegas(i)));
    ylabel('h(t)')
    xlabel('t')
   
    subplot(2,1,2)
    root = roots([1 2*zeta*omegas(i) omegas(i)^2]);
    plot(real(root), imag(root), 'o')
    title("poles")
    ylabel("Imaginary Part")
    xlabel("Real Part")
    hold on
    shg
    pause(0.1);
end
hold off

