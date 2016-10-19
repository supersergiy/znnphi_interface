CPUs = [ 1 2 4 8 16 32 64 ];
HT1  = [ 5.18 2.6 1.3 0.65 0.327 0.164 0.083];
HT2  = [ 4.81 2.4 1.2 0.6 0.3 0.153 0.082];
HT4  = [ 4.685 2.34 1.17 0.586 0.295 0.152 0.079];

Util = (100 * 236760072192 ./ (CPUs .* HT4 * 1.1 * 64 * 1000000000));

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.465999990701675 0.674000024795532 0.187999993562698;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628;0.850000023841858 0.324999988079071 0.0979999974370003],...
    'FontSize',12);
box(axes1,'on');
hold(axes1,'on');

% Create ylabel
xlabel('Cores used');

% Create xlabel
ylabel('Images per second');

plot(CPUs,1./HT1,'DisplayName','1 Hardware Thread','LineWidth',3);
plot(CPUs,1./HT2,'DisplayName','2 Hardware Thread','LineWidth',3);
plot(CPUs,1./HT4,'DisplayName','4 Hardware Thread','LineWidth',3);


% Create axes
axes2 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'HitTest','off','Color','none',...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0.465999990701675 0.674000024795532 0.187999993562698;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628;0.850000023841858 0.324999988079071 0.0979999974370003],...
    'FontSize',12,...
    'YLim', [0 100],...
    'YAxisLocation','right');
box(axes2,'on');
hold(axes2,'on');

% Create xlabel
ylabel('Utilization (%)');

plot(CPUs,Util,'DisplayName','1 Hardware Thread','LineWidth',3);
