# Routine to download automatically SMN and gauge data from the DWD using their R library
# Daniel Wolfensberger, LTE - EPFL / MeteoSwiss, 2020
# Rebecca Gugerli, LTE-EPFL/ MeteoSwiss, 2022
# The routine is meant to be called in command line
# Rscript retrieve_dwh_data.r t0 t1 threshold stations variables output_folder missing_value overwrite
# t0 : start time in YYYYMMDDHHMM format
# t1 : end time in YYYYMMDDHHMM format
# threshold : minimum value of hourly precipitation for the entire hour to be included in the database (i.e. all 6 10min timesteps)
# stations : list of station abbrevations, e.g. "PAY, GVE, OTL"
# variables : list of variables to retrieve, using the DWH names, for example "tre200s0,prestas0,ure200s0,rre150z0,dkl010z0,fkl010z0"
# output_folder : directory where to store the csv files with the retrieved data
# overwrite : whether or not to overwrite already existing data in the output_folder
  
# IMPORTANT: this function is called by the main routine in database.py
# so you should never have to call it manually


.libPaths("/store/msclim/share/CATs/cats/lib-R3.5.2/")
library('mchdwh')
.libPaths()
library('R.utils')
library('plyr')
library('lubridate')

# This function adds new data to a file, with/without overwriting already present data
append_to_file <- function(fname, df, overwrite = FALSE){
  if(!file.exists(fname)){
    df_merged = df
  }
  else{
    df_old = data.table::fread(fname, sep = ',')
    if(!overwrite){
      not_in_old = !df$TIMESTAMP %in% df_old$TIMESTAMP
      df_merged = bind_rows(df_old, df[not_in_old,])
    }
    else{
      not_in_new = !df_old$TIMESTAMP %in% df$TIMESTAMP
      df_merged = data.table::rbindlist(list(df_old[not_in_new,],df), fill = T)
    }
    df_merged = df_merged[order(df_merged$TIMESTAMP),]
  }
  return(as.data.frame(df_merged))
}
# 
# Parsing command line arguments
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
script.basename <- paste(dirname(script.name),'/',sep = '')
print(initial.options)
args <-commandArgs(TRUE)

t0_str <- args[1]
tend_str <- args[2]
threshold <- args[3]
stations <- args[4]
variables <- args[5]
output_folder <- args[6]
missing_value <- as.numeric(args[7])
overwrite <- as.numeric(args[8])
variables = gsub("[()]","",variables)
variables = strsplit(variables,',')[[1]]
stations = gsub("[()]","",stations)
stations = strsplit(stations,',')[[1]]

# #For testing
# t0_str <- '201810300000'
# tend_str <- '201810301200'
# threshold <- 0.1
# stations = c('ARO','LSN')
# variables = "rre150z0,tre200s0,fkl010z0,rre150z0_adj"
# variables = gsub("[()]","",variables)
# variables = strsplit(variables,',')[[1]]
# output_folder = '/scratch/rgugerli/rainforest_debug/dbase_tests/'
# script.basename = '/scratch/rgugerli/rainforest_debug/rainforest/rainforest/'
# missing_value <- -9999
# overwrite <- TRUE

# Get directory of data_stations.csv, i.e rainforest/common/constants
# station_info_path = file.path(dirname(script.basename), 'common', 'data', 'data_stations.csv')
path <- Sys.getenv("RAINFOREST_DATAPATH")
station_info_path = file.path(path, 'references', 'metadata', 'data_stations.csv' )
station_info = data.table::fread(station_info_path, sep = ';')
station_info <- as.data.frame(station_info)

for(i in 1:length(stations)){
  print(paste('Retrieving station ',stations[i]))
  isvalid = TRUE
  variables_var <- variables
  if(!isvalid){
    next
  }
  
  tryCatch({
      data <- dwhget_surface(nat_abbr = stations[i], param_short = variables_var, date = c(t0_str,tend_str))
      # Reorganize data into multidimensional array
      data <- reshape_data(x=data, reshape="stp")

      # Assume transfer function (catch efficiency) is possible
      CE <- TRUE
      
      # Fill up missing columns
      for(j in 1:length(variables)){
        if(!(variables[j] %in% colnames(data))){
          data[variables[j]] = missing_value
          # Add a tag for CatchEfficiency
          if(variables[j] == 'fkl010z0'){
          CE <- FALSE
          data['rre150z0_adj'] = missing_value
          print(paste('Transfer function not applicable for ',stations[i],sep = ''))
          }
        }
      }
      

      if(CE){
        data['rre150z0_adj'] <- data['rre150z0']
        tryCatch({
          print(paste('Adjustment of gauge measurement ', stations[i]))
          # Add Kochendorfer Equation (see https://hess.copernicus.org/articles/21/3525/2017/hess-21-3525-2017.pdf)
          wind <- data[["fkl010z0"]]
          wind[wind > 9] <- 9.
          # Mixed precipitation (-2<=tair<=2)
          a_mixed <- 0.624
          b_mixed <- 0.185
          c_mixed <- 0.364
          CEmixed_KD4 <- a_mixed * exp(-b_mixed * wind)  + c_mixed
          index_mixed <- (data$tre200s0 >= -2) & (data$tre200s0 <= 2)  & (!is.na(data$tre200s0))
          data$rre150z0_adj[index_mixed] <- data$rre150z0[index_mixed] / CEmixed_KD4[index_mixed]
          # solid precipitation (<-2)
          a_solid <- 0.865
          b_solid <- 0.298
          c_solid <- 0.225
          CEsolid_KD4 <- a_solid * exp(-b_solid * wind)  + c_solid
          index_solid <- (data$tre200s0 < -2) & (!is.na(data$tre200s0))
          data$rre150z0_adj[index_solid] <- data$rre150z0[index_solid] / CEsolid_KD4[index_solid]
          data$rre150z0_adj=signif(data$rre150z0_adj, digits = 3)
          }, error=function(e){print(e);print(paste('Catch efficiency failed for ',stations[i],sep = ''))})
        }
      else {
          data['rre150z0_adj'] = missing_value
      }

      # Time of beginning of measurement
      tstamps <- as.POSIXct(strptime(data$datetime,'%Y%m%d%H%M',tz='UTC')) - 60 * 5
      stahours_all <- paste(data[["nat_abbr"]],format(ceiling_date(tstamps,'hour'), format="%Y%m%d%H"))
      data[["stahours"]] = stahours_all
      
      # Get all wet hours
      rain_h <- aggregate(data[["rre150z0"]], by=list(data[["stahours"]]), FUN=sum, na.rm=TRUE)
      hours_wet <- rain_h[["x"]] >= threshold
      stahours_wet <- rain_h[["Group.1"]][hours_wet]
      
      zhours_wet <- stahours_all %in% stahours_wet
      zdata_wet <- data[zhours_wet,c('datetime',variables)]

      name_out <- paste(output_folder, '/', stations[i], '.csv.gz', sep = '')
      # date format must be changed to linux timestamp
      zdata_wet$datetime = as.numeric(as.POSIXct(strptime(zdata_wet$datetime,'%Y%m%d%H%M',tz='UTC')))
      colnames(zdata_wet)[1] = 'timestamp'
      
      # Finally since the R format of hdf5 is not compatible with dask and pandas
      # I choose to stay with csv
      zdata_wet[is.na(zdata_wet)] = missing_value
      
      # Add station info
      zdata_wet <- cbind(station = stations[i], zdata_wet)

      # Reorder if needed to always get same col order
      right_order = c('station','timestamp', variables)
      zdata_wet = zdata_wet[,right_order]
      colnames(zdata_wet) <- toupper(colnames(zdata_wet))
      
      zdata_wet = append_to_file(name_out, zdata_wet, overwrite)
      write.table(zdata_wet, gzfile(name_out), sep=',',col.names = T, row.names = F)
      
    }, error=function(e){print(e);print(paste('Data not available for ',stations[i],sep = ''))})
  }

