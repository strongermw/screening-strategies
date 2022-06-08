#format <- ".pdf"
date <- "sim1000_0606"
setwd("D:/FDU program/COVID-19/研究 筛查检测策略/可视化/v10 可视化")
library(dplyr)
library(ggplot2)
library(reshape2)
library(ggsci)
library(RColorBrewer)
library(stringr)
library(scales)
library(cowplot)
library(ggpubr)
### data import ################################################################

data <- read.csv("compartments_time series_sim1000_0606.csv")
data1 <- read.csv("cumulative & maximum data_sim1000_0606.csv")
rt <- read.csv('Rt_sim1000_0606.csv')

stra_list <- read.csv("Strategy&Scenario_0608.csv")
data <- merge(data,stra_list,by=c("stra_file","scen_file"),all.x=T)



#scenario order
dt_nbat <- data %>% filter(!is.na(order_nonbatch_0)) %>% arrange(order_nonbatch_0)
dt_nbat$scen_name <- factor(dt_nbat$scen_name,levels=unique(dt_nbat$scen_name))

dt_nbat_dis <- data %>% filter(!is.na(order_nonbatch_distri)) %>% arrange(order_nonbatch_distri)
dt_nbat_dis$scen_distri_name <- factor(dt_nbat_dis$scen_distri_name,levels=unique(dt_nbat_dis$scen_distri_name))

dt_bat <- data %>% filter(!is.na(order_batch)) %>% arrange(order_batch)
dt_bat$scen_name <- factor(dt_bat$scen_name,levels=unique(dt_bat$scen_name))

dt_bat_dis <- data %>% filter(!is.na(order_batch_distri)) %>% arrange(order_batch_distri)
dt_bat_dis$scen_distri_name <- factor(dt_bat_dis$scen_distri_name,levels=unique(dt_bat_dis$scen_distri_name))

dt_bat$scen_name <- sub("1-weekly","1 batch-weekly",dt_bat$scen_nam)


#### Fig 1   time series ######################################################

## Fig 1-A
dt_1A <- dt_nbat %>% filter(stra_name=="Full")

dt_1A$ifstop <- ifelse(dt_1A$infected_m==0,1,0)
dt_1A_stop <- dt_1A[!duplicated(dt_1A[,c("scen_name","ifstop")]),]
dt_1A_stop <- dt_1A_stop %>% filter(ifstop==1)
dt_1A_stop0 <- dt_1A[!duplicated(dt_1A[,c("scen_name")]),]
dt_1A_stop0 <- dt_1A_stop0 %>% subset(select="scen_file")
dt_1A_stop <- dt_1A_stop %>% merge(dt_1A_stop0,by="scen_file",all=T) %>%
  arrange(order_nonbatch_0)
dt_1A_stop$scen_name <- factor(dt_1A_stop$scen_name,levels=unique(dt_1A_stop$scen_name))


colors <- brewer.pal(11,"Spectral")[11:1]
colors <- c(colors[1:2],"DeepSkyBlue",colors[3],"YellowGreen","yellow2",colors[7:11])


title <- paste0("Current Number of Infectors","  -  ","Full")

fig1Am <- ggplot(dt_1A,aes(x=time))+
  geom_ribbon(aes(ymin=infected_25,ymax=infected_75,fill=scen_name),alpha = 0.06)+
  scale_fill_manual(values=colors)+
  
  geom_line(aes(y=infected_m,group=scen_name,color=scen_name))+
  scale_color_manual(values=colors)+
  #geom_vline(data=dt_1A_stop,aes(xintercept=time),size=0.5,lty=2,alpha = 0.7,color=colors)+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5))+
  guides(color=guide_legend(ncol=5))+
  labs(x = "time", y = "counts", title = title, tag="A") ;fig1Am


dt_1As <- dt_1A %>% filter(order_nonbatch_0==c(7:11)) %>% filter(time<=50)

fig1As <- ggplot(dt_1As,aes(x=time))+
  #geom_ribbon(aes(ymin=infected_25,ymax=infected_75,fill=scen_name),alpha = 0.06)+
  scale_fill_manual(values=colors[7:11])+
  
  geom_line(aes(y=infected_m,group=scen_name,color=scen_name))+
  scale_color_manual(values=colors[7:11])+
  
  theme_classic()+
  theme(legend.position = "none"
        #,rect = element_rect(fill = "transparent",color="transparent")
  )+
  labs(x = "time", y = "counts") ;fig1As


fig1A <- ggdraw()+
  draw_plot(fig1Am)+
  draw_plot(fig1As,scale=0.35,hjust=-0.25,vjust=-0.01) ;fig1A



##Fig 1-B

rt <- rt %>% merge(stra_list,by="scen_file",all.x=T) %>%
  filter(time <=70) %>% 
  filter(!is.na(order_nonbatch_0)) %>%
  arrange(order_nonbatch_0)
rt$scen_name <- factor(rt$scen_name,levels=unique(rt$scen_name))


colors <- brewer.pal(11,"Spectral")[11:1]
colors <- c(colors[1:2],"DeepSkyBlue",colors[3],"YellowGreen","yellow2",colors[7:11])

title <- paste0("Effective Reproduction Number","  -  ","Full")
table(rt$scen_name)

fig1B <- ggplot(rt,aes(x=time))+
  geom_ribbon(aes(ymin=lower,ymax=upper,fill=scen_name),alpha = 0.01)+
  scale_fill_manual(values=colors)+
  
  geom_line(aes(y=Rt,group=scen_name,color=scen_name))+
  scale_color_manual(values=colors)+
  
  geom_hline(yintercept=1,color="black",size=0.5,lty=2)+ 
  
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5))+
  guides(color=guide_legend(ncol=5))+
  labs(x = "time", y = "counts", title = title, tag="B") ;fig1B






##Fig 1-C  

# 4 batches-weekly   order_batch_distri<=4)
# 3 batches-weekly   order_batch_distri>4)

#
num_batch <- 3
dt_1C <- data %>% filter(stra_name==c("Full")) %>% 
  filter(order_batch_distri>4|order_nonbatch_distri>4)
dt_1C_com <-dt_1C  %>%
  melt(id.vars=c("stra_name","scen_name","scen_distri_name","time"),
       measure.vars=c("E_m","I_pre_m","I_sym_m","I_asym_m","Q_E_m","Q_pre_m","Q_sym_m","Q_asym_m","H_m"),
       variable.name="compartment",
       value.name="counts") 
#%>%
#filter(time<=65)
dt_1C_com$compartment <- sub(pattern="_m",replacement="",dt_1C_com$compartment)
dt_1C_com$compartment <- sub(pattern="_",replacement="",dt_1C_com$compartment)
dt_1C_com$compartment <- factor(dt_1C_com$compartment,levels=c("E","QE","Ipre","Qpre","Isym","Qsym","Iasym","Qasym","H"))
dt_1C_com$scen_distri_name <- factor(dt_1C_com$scen_distri_name,levels=c("even (day 147)","beginning (day 123)","middle (day 345)","end (day 567)"))

title <- paste0("Current number of individuals in compartments","  -  ","Different Times")
fig1C <- ggplot(dt_1C_com,aes(x=time))+
  geom_area(aes(y=counts,fill=compartment),alpha=0.5)+
  #scale_fill_simpsons()+
  scale_fill_lancet()+
  facet_grid(~scen_distri_name)+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_legend(nrow=1))+
  labs(x = "time", y = "counts", title = title, tag="C");fig1C


colors <- pal_lancet()(4)
ggplot(dt_1C,aes(x=time))+
  geom_ribbon(aes(ymin=infected_25,ymax=infected_75,fill=scen_distri_name),alpha = 0.06)+
  scale_fill_manual(values=colors)+
  facet_grid(stra_name~scen_distri_name)+
  geom_line(aes(y=infected_m,group=scen_distri_name,color=scen_distri_name))+
  scale_color_manual(values=colors)+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5))+
  labs(x = "time", y = "counts", title = title) 





## Fig 1-D

dt_1D <- dt_bat %>% filter(stra_name=="Full in Batches"|stra_name=="Full")
dt_1D$ifstop <- ifelse(dt_1D$infected_m==0,1,0)
dt_1D_stop <- dt_1D[!duplicated(dt_1D[,c("scen_name","ifstop")]),]
dt_1D_stop <- dt_1D_stop %>% filter(ifstop==1)
dt_1D_stop0 <- dt_1D[!duplicated(dt_1D[,c("scen_name")]),]
dt_1D_stop0 <- dt_1D_stop0 %>% subset(select="scen_file")
dt_1D_stop <- dt_1D_stop %>% merge(dt_1D_stop0,by="scen_file",all=T) %>%
  arrange(order_batch)
dt_1D_stop$scen_name <- factor(dt_1D_stop$scen_name,levels=unique(dt_1D_stop$scen_name))



colors <-colorRampPalette(c("yellow2","red4"))(7)
title <- paste0("Current Number of Infectors","  -  ","Full in Batches")

fig1D <- ggplot(dt_1D,aes(x=time))+
  geom_line(aes(y=infected_m,group=scen_name,color=scen_name))+
  scale_color_manual(values=colors)+  
  
  geom_ribbon(aes(ymin=infected_25,ymax=infected_75,fill=scen_name),alpha = 0.06)+
  scale_fill_manual(values=colors)+
  #geom_vline(data=dt_1D_stop,aes(xintercept=time),size=0.5,lty=2,alpha = 0.7,color=colors)+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5))+
  guides(color=guide_legend(ncol=3))+
  labs(x = "time", y = "counts", title = title, tag="D");fig1D



#合并
ggarrange(fig1A,fig1B,fig1C,fig1D,nrow=2,ncol=2)
ggsave(paste0("./Plots/","Fig2 time series","_",date,format),width = 14,height = 10,dpi = 500)




