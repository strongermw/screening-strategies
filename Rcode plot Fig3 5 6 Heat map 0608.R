#format <- ".png"
date<- "sim1000_0606"
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

data <- read.csv("cumulative & maximum data_sim1000_0606.csv")


stra_list <- read.csv("Strategy&Scenario_0608.csv")
data <- merge(stra_list,data,by=c("stra_file","scen_file"),all.x=T)

data$stra_name <- sub(" in Batches","",data$stra_name)

#scenario order
dt_nbat <- data %>% filter(!is.na(order_nonbatch)) %>% arrange(order_nonbatch)
dt_nbat$scen_name <- factor(dt_nbat$scen_name,levels=unique(dt_nbat$scen_name))


dr_nbat <- dt_nbat
for (i in 1:ncol(dr_nbat)){
  if (is.numeric(dr_nbat[,i])) {dr_nbat[,i] <- round(dr_nbat[,i],0)
  } else {dr_nbat[,i] <- dr_nbat[,i]}
}

dt_bat <- data %>% filter(!is.na(order_batch)) %>% arrange(order_batch)
dt_bat$scen_name <- factor(dt_bat$scen_name,levels=unique(dt_bat$scen_name))

#dt_bat$stra_name <- ifelse(dt_bat$scen_name=="1-weekly",
                           #paste0(dt_bat$stra_name," in Batches"),
                           #dt_bat$stra_name)
dt_bat$scen_name <- sub("1-weekly","1 batch-weekly",dt_bat$scen_nam)
dr_bat <- dt_bat
for (i in 1:ncol(dr_bat)){
  if (is.numeric(dr_bat[,i])) {dr_bat[,i] <- round(dr_bat[,i],0)
  } else {dr_bat[,i] <- dr_bat[,i]}
}

#### Fig 3    ######################################################




#Proportion of cumulative infectors
title <- paste0("Cumulative Infection Rate (%)","  -  ","Non-batch")
fig2A1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=proportion_infected_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "Thistle1",high =  "MediumPurple4")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title,tag="A1")+
  geom_text(data=dr_nbat,aes(x=stra_name,y=scen_name,
                label=paste0(proportion_infected_sum_m,
                             "(",proportion_infected_sum_25,",",
                             proportion_infected_sum_75,")")));fig2A1

title <- paste0("Cumulative Infection Rate (%)","  -  ","in Batches")
fig2A2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=proportion_infected_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "Thistle1",high =  "MediumPurple4")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title,tag="A2")+
  geom_text(data=dr_bat,aes(x=stra_name,y=scen_name,
                label=paste0(proportion_infected_sum_m,
                             "(",proportion_infected_sum_25,",",
                             proportion_infected_sum_75,")")));fig2A2



#Proportion of infectors missed
title <- paste0("Miss Rate(%)","  -  ","Non-batch")
fig2B1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=proportion_infected_miss_m))+
  geom_raster()+
  scale_fill_gradient(low = "LavenderBlush2",high =  "HotPink4")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title, tag="B1")+
  geom_text(data=dr_nbat,aes(x=stra_name,y=scen_name,
                label=paste0(proportion_infected_miss_m,
                             "(",proportion_infected_miss_25,",",
                                         proportion_infected_miss_75,")")));fig2B1

title <- paste0("Miss Rate (%)","  -  ","in Batches")
fig2B2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=proportion_infected_miss_m))+
  geom_raster()+
  scale_fill_gradient(low = "LavenderBlush2",high =  "HotPink4")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title, tag="B2")+
  geom_text(data=dr_bat,aes(x=stra_name,y=scen_name,
                label=paste0(proportion_infected_miss_m,
                             "(",proportion_infected_miss_25,",",
                             proportion_infected_miss_75,")")));fig2B2



#Tests per positive
title <- paste0("Tests per positive","  -  ","Non-batch")
fig2C1 <-ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=tested_per_positive_m))+
  geom_raster()+
  scale_fill_gradient(low = "Wheat1",high =  "Firebrick")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title,tag="C1")+
  geom_text(data=dr_nbat,aes(x=stra_name,y=scen_name,
                            label=paste0(tested_per_positive_m,
                                         "(",tested_per_positive_25,",",
                                         tested_per_positive_75,")")));fig2C1


title <- paste0("Tests per positive","  -  ","in Batches")
fig2C2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=tested_per_positive_m))+
  geom_raster()+
  scale_fill_gradient(low = "Wheat1",high =  "Firebrick")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title,tag="C2")+
  geom_text(data=dr_bat,aes(x=stra_name,y=scen_name,
                label=paste0(tested_per_positive_m,
                                     "(",tested_per_positive_25,",",
                                                 tested_per_positive_75,")")));fig2C2



ggarrange(fig2A1,fig2B1,fig2C1,fig2A2,fig2B2,fig2C2,ncol=3,nrow=2)

ggsave(paste0("./Plots/","Fig3 disease control and efficiency heat map","_",date,format),width = 20,height = 10,dpi = 500)




#### Fig 5    ######################################################

#Cumulative of Q hotel
title <- paste0("Cumulative hotel quarantined people","  -  ","Non-batch")
fig3A1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=Q_hotel_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "MistyRose2",high =  "Maroon3")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title, tag="A1")+
  geom_text(data=dr_nbat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_hotel_sum_m,"(",Q_hotel_sum_25,",",Q_hotel_sum_75,")")));fig3A1


title <- paste0("Cumulative hotel quarantined people","  -  ","in Batches")
fig3A2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=Q_hotel_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "MistyRose2",high =  "Maroon3")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title, tag="A2")+
  geom_text(data=dr_bat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_hotel_sum_m,"(",Q_hotel_sum_25,",",Q_hotel_sum_75,")")));fig3A2


#Cumulative of Q shelter
title <- paste0("Cumulative shelter quarantined cases","  -  ","Non-batch")
fig3B1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=Q_shelter_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "Cornsilk",high =  "Orange")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title, tag="B1")+
  geom_text(data=dr_nbat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_shelter_sum_m,"(",Q_shelter_sum_25,",",Q_shelter_sum_75,")")));fig3B1


title <- paste0("Cumulative shelter quarantined cases","  -  ","in Batches")
fig3B2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=Q_shelter_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "Cornsilk",high =  "Orange")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title, tag="B2")+
  geom_text(data=dr_bat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_shelter_sum_m,"(",Q_shelter_sum_25,",",Q_shelter_sum_75,")")));fig3B2


#Cumulative of H
title <- paste0("Cumulative hostipalized cases","  -  ","Non-batch")
fig3C1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=H_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "AntiqueWhite1",high =  "Brown4")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title, tag="C1")+
  geom_text(data=dr_nbat,
            aes(x=stra_name,y=scen_name,
                label=paste0(H_sum_m,"(",H_sum_25,",",H_sum_75,")")));fig3C1

title <- paste0("Cumulative hostipalized cases","  -  ","in Batches")
fig3C2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=H_sum_m))+
  geom_raster()+
  scale_fill_gradient(low = "AntiqueWhite1",high =  "Brown4")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title, tag="C2")+
  geom_text(data=dr_bat,
            aes(x=stra_name,y=scen_name,
                label=paste0(H_sum_m,"(",H_sum_25,",",H_sum_75,")")));fig3C2





ggarrange(fig3A1,fig3B1,fig3C1,fig3A2,fig3B2,fig3C2,
          ncol=3,nrow=2)
ggsave(paste0("./Plots/","Fig5 rescorce heat map","_",date,format),width = 20,height = 10,dpi = 500)




#### fig 6 ##########################################################

#Maximum of hotel quarantined people
title <- paste0("Maximum of hotel quarantined people","  -  ","Non-batch")

fig6A1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=Q_hotel_max_m))+
  geom_raster()+
  scale_fill_gradient(low = "lavender",high =  "MediumPurple2")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title, tag="A1")+
  geom_text(data=dr_nbat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_hotel_max_m,"(",Q_hotel_max_25,",",Q_hotel_max_75,")")));fig6A1


title <- paste0("Maximum of hotel quarantined people","  -  ","in Batches")
fig6A2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=Q_hotel_max_m))+
  geom_raster()+
  scale_fill_gradient2(low = "lavender",high =  "MediumPurple2")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title, tag="A2")+
  geom_text(data=dr_bat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_hotel_max_m,"(",Q_hotel_max_25,",",Q_hotel_max_75,")")));fig6A2

#Maximum of shelter quarantined people
title <- paste0("Maximum of shelter quarantined cases","  -  ","Non-batch")
fig6B1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=Q_shelter_max_m))+
  geom_raster()+
  scale_fill_gradient(low = "Cornsilk1",high =  "Chocolate2")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title, tag="B1")+
  geom_text(data=dr_nbat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_shelter_max_m,"(",Q_shelter_max_25,",",Q_shelter_max_75,")")));fig6B1


title <- paste0("Maximum of shelter quarantined cases","  -  ","in Batches")
fig6B2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=Q_shelter_max_m))+
  geom_raster()+
  scale_fill_gradient(low = "Cornsilk1",high =  "Chocolate2")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title, tag="B2")+
  geom_text(data=dr_bat,
            aes(x=stra_name,y=scen_name,
                label=paste0(Q_shelter_max_m,"(",Q_shelter_max_25,",",Q_shelter_max_75,")")));fig6B2



#Maximum of hostipalized cases 
title <- paste0("Maximum of hostipalized cases","  -  ","Non-batch")
fig6C1 <- ggplot(dt_nbat,aes(x=stra_name,y=scen_name,fill=H_max_m))+
  geom_raster()+
  scale_fill_gradient(low = "Seashell",high =  "Firebrick1")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="frequency",title = title, tag="C1")+
  geom_text(data=dr_nbat,
            aes(x=stra_name,y=scen_name,
                label=paste0(H_max_m,"(",H_max_25,",",H_max_75,")")));fig6C1

title <- paste0("Maximum of hostipalized cases","  -  ","in Batches")
fig6C2 <- ggplot(dt_bat,aes(x=stra_name,y=scen_name,fill=H_max_m))+
  geom_raster()+
  scale_fill_gradient(low = "Seashell",high =  "Firebrick1")+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        strip.background = element_blank())+
  guides(fill=guide_colourbar(title.position="top",barwidth=10,barheight=0.8))+
  labs(x="sampling strategy",y="batch",title = title, tag="C2")+
  geom_text(data=dr_bat,
            aes(x=stra_name,y=scen_name,
                label=paste0(H_max_m,"(",H_max_25,",",H_max_75,")")));fig6C2


ggarrange(fig6A1,fig6B1,fig6C1,fig6A2,fig6B2,fig6C2,
          ncol=3,nrow=2)
ggsave(paste0("./Plots/","Fig6 maximum rescorce heat map","_",date,format),width = 20,height = 10,dpi = 500)

