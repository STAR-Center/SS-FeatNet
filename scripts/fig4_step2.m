algos = {'sample2','not100lr3jit2','3dfeatnetkp_sample','3dfeatnetkp_not100lr3jit','learnDesc_1key_256','learnDesc_1key_3df_256'};%,'fpfh'，'not100lr3jitAttention','not100lr5jitAttention70epoch','not100','not100lr3jit05','not100lr3jit','fpfh','not100lr5jit70epoch'};
legends = {'ISS+3DFeatNet desc','ISS+our desc','3DF kpt+3DFeatNet desc','3DF kpt+our desc','ISS kpt+ our ISS desc','3DF kpt + our 3DF desc'};%,'ISS+fpfh'，'not100lr3jitAttention','not100lr5jitAttention70epoch','not100','not100lr3jit05','not100lr3jit','fpfh','not100lr5jit70epoch'};

distances = 0.1 : 0.1 : 10;

numAlgos = length(algos);
precisions = {};
algoNames = {};

for iAlgo = 1 : numAlgos
    algoName = algos{iAlgo};
    load(fullfile('results_oxford', sprintf('matching_statistic-%s.mat', algoName)));
    
    distTable = cat(1, statisticTable.nearestMatchDist{:});
    
    precisionTable = zeros(size(distances));
    for iDist = 1 : length(distances)
        precisionTable(iDist) = nnz(distTable < distances(iDist));
    end
    precisionTable = precisionTable/length(distTable);
    
    precisions{iAlgo} = precisionTable * 100;
    algoNames{iAlgo} = algoName;
end

%%
figure(1), clf, hold on
for iAlgo = 1 : numAlgos
    if iAlgo > 4
        plot(distances, precisions{iAlgo}, '--','linewidth',2);
    else
        plot(distances, precisions{iAlgo},'linewidth',2);
    end
end

ylim([0 55])
yl = ylim;
plot([1, 1], yl, '--k');

% legend(legends, 'Location', 'Northwest'), 'EastOutside'
legend(legends, 'Location','Southeast','FontSize',20)
ylabel('Precision (%)','FontSize',20);
xlabel('Meters','FontSize',20) 
