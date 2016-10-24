g_legend <- function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

do_plot <- function(a.net,a.pas,a.cpu,a.flops) {
  fname <- paste("/Users/zlateski/Dropbox/MIT/ZNNphi/benchmarks/",
                 a.cpu,
                 "/",
                 a.net,
                 "-",
                 a.pas,
                 ".csv", sep="")
  d <- read.csv(fname)
  data <- aggregate(list(Utilization = d$GFLOPs.s/a.flops), 
                    by=list(Cores = d$Cores, Layer = d$Layer), 
                    FUN=max)
  p <- ggplot(data,
              aes(x=Cores,
                  y=Utilization,
                  color=Layer)) + 
    geom_point() + geom_line()

  leg <- g_legend(p)
  
  legfname <- paste("/Users/zlateski/Dropbox/MIT/ZNNphi/paper/fig/",
                    a.net,
                    "-legend.pdf", sep="")
  
  pdf(legfname,width=0.6,height=2)
  grid.arrange(leg,ncol=1)
  dev.off()

  plotfname <- paste("/Users/zlateski/Dropbox/MIT/ZNNphi/paper/fig/",
                     a.net,
                     "-", a.pas, "-", a.cpu,
                     ".pdf", sep="")
  
  pdf(plotfname,width=2,height=1.5)

  zz <- ggplot(data,
            aes(x=Cores,
                y=Utilization,
                color=Layer))+
  geom_point() + geom_line() + theme(legend.position = "none", 
                                     axis.title.x=element_blank(),
                                     axis.title.y=element_blank()) +
  coord_cartesian(ylim=c(0, 1))
  
  grid.arrange(zz,ncol=1)
  
  
  dev.off()
  }

do_plot('vgg', 'upd', 'skylake', 512)
do_plot('vgg', 'fwd', 'skylake', 512)

do_plot('vgg', 'upd', 'haswell', 2.5*72*32)
do_plot('vgg', 'fwd', 'haswell', 2.5*72*32)

do_plot('vgg', 'upd', 'knl', 1.1*64*64)
do_plot('vgg', 'fwd', 'knl', 1.1*64*64)


do_plot('unet', 'upd', 'skylake', 512)
do_plot('unet', 'fwd', 'skylake', 512)

do_plot('unet', 'upd', 'haswell', 2.5*72*32)
do_plot('unet', 'fwd', 'haswell', 2.5*72*32)

do_plot('unet', 'upd', 'knl', 1.1*64*64)

do_plot('d3d', 'upd', 'skylake', 512)
do_plot('d3d', 'fwd', 'skylake', 512)

do_plot('d3d', 'upd', 'haswell', 2.5*72*32)
