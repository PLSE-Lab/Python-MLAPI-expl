#!/usr/bin/env python
# coding: utf-8

# **<font size=6>Shiny Application:</font>**
# <br>
# I've created an R shiny web application using modified speed camera data. I'm not sure if I am able to run a shiny application within the notebook so I've simply included a link to the application below. It seems to work best using Chrome.
# 
# **Link:** https://cole-soffa.shinyapps.io/ChicagoSpeedCameras/
# 
# I've included a couple of static images of plot shown within the Shiny application. Because they are static images, I've included additional information (legends, annotation, etc.) to help the viewer understand the graphs easier. Within the actual application it is a little bit more self-explanatory.

# In[ ]:


from IPython.display import Image
Image("../input/monthly-speed-violations-image/monthly_speed_violations_both.jpg")


# In[ ]:


from IPython.display import Image
Image("../input/camera-violations-image/camera_violations_both.jpg")


# **<font size=6>Source Code:</font>**
# <br>
# Displayed below is the source code for the Shiny web application. It is **R code** not Python code. I made the notebook in Python because it was easiest to render images within the kernel using Python.
# 
# Please feel free to comment or message with any questions about my approach! If you've viewed the Shiny application, you can see that there are some minor formatting/spacing issues between plots. I haven't found a solution to these bugs so if you have any advice, it would be appreciated.

# In[ ]:


library(shiny)
library(dplyr)
library(readr)
library(ggplot2)
library(RColorBrewer)
library(ggiraph)
library(devtools)
library(ggmap)
library(scales)
library(ggrepel)

# First file to read in, contains all of the speed violations
modified_speed_violations <- read_csv("Modified_speed_violations.csv", col_types = cols(Address = col_skip(), X1 = col_skip()))
modified_speed_violations <- mutate(modified_speed_violations, Month = substr(Month_Day, 1, 2))

months <- c('01' = 'January', '02' = 'February', '03' = 'March', '04' = 'April', '05' = 'May', '06' = 'June', '07' = 'July', '08' = 'August', '09' = 'September', '10' = 'October', '11' = 'November', '12' = 'December')
modified_speed_violations$Month_name = months[modified_speed_violations$Month]

#-------------------------------------------------------------------------------------------------------------------
# Total speed violations by month each year and for each type (school or park zone)
monthly_violations_by_type <- modified_speed_violations %>%
  group_by(Year, Month, Month_name, Type) %>%
  summarise(
    Total_violations = sum(VIOLATIONS)
  )

#-------------------------------------------------------------------------------------------------------------------
# Calculates the average number of speed violations each month for each year
mean_grouped <- function(df, group_var1, group_var2, summ_var) {
  group_var1 = enquo(group_var1)
  group_var2 = enquo(group_var2)
  summ_var = enquo(summ_var)
  df %>% group_by(UQ(group_var1), UQ(group_var2)) %>%
    summarise(
      sum = sum(UQ(summ_var))
    ) %>%
    summarise(
      m = mean(sum)
    )
}
monthly_mean <- mean_grouped(monthly_violations_by_type, Year, Month, Total_violations)

#-------------------------------------------------------------------------------------------------------------------
# Establish a google map in the Chicago area
google_key <- "*Insert google API key here*"
register_google(google_key)
map <- get_map(location = c(lon=-87.6298, lat=41.8781), zoom=11, maptype='roadmap', source='google')

#-------------------------------------------------------------------------------------------------------------------
# Define ui for shiny app
ui <- fluidPage(
  tags$h2("An analysis of speed camera violations in the city of Chicago"),
  ("The City of Chicago has published speed camera violations data dating back to 2014. This data is used to build an interactive web application that guides users through various angles analyzing speed cameras. Each angle uses a different visualization technique to try to answer the following questions below. Hovering and clicking on different data points from each plot will reveal additional information and interactive features. "),
  hr(),
  # First row of plots
  fluidRow(
    column(
      width = 6,
      tags$h4("Have speed cameras been effective since being installed?"),
      HTML("<b>", "Note: ", "</b>", "Hover over and select any month to reveal additional information"),
      ('/n'),
      ggiraphOutput("plot"),
      tags$em(htmlOutput("text", width = "100%", align = 'center'))
    ),
    column(
      width = 6,
      tags$h4("How do speed violations vary seasonally? Are there consistent trends throughout the year?"),
      HTML("<b>", "Note: ", "</b>", "Hover over and select circled months to reveal additional information"),
      selectInput("type", "Type:", c("Park", "School")),
      ggiraphOutput("plot1"),
      tags$em(htmlOutput("text1", width = "100%", align = 'center'))
    )
  ),
  hr(),
  # Second row of plots
  fluidRow(
    column(
      tags$h4("Are there specific cameras that conistently have more violations than others? If so, where are they located?"),
      HTML("<b>", "Note: ", "</b>", "Each data point represents a specific speed camera. Hover and select to reveal its location in Chicago. Move the slider below the plot to show how each speed camera has varied since 2014"),
      width = 6,
      selectInput("type1", "Type:", c("Park", "School")),
      actionButton("reset", label = "Reset selected cameras"),
      ggiraphOutput("plot2"),
      sliderInput("year",
                  "Year",
                  min = 2014,
                  max = 2018,
                  value = 2014,
                  step = 1,
                  width = "100%",
                  animate = animationOptions(interval = 1500, loop = TRUE),
                  sep = ''),
      tags$em(htmlOutput("text2", width = "100%", align = 'center'))
    ),
    column(width = 6,
           ggiraphOutput("plot3", width = "100%"),
           htmlOutput("text3", width = "100%", align = 'center')
    )
  ),
  ("Cole Soffa"),
  tags$br(),
  tags$em("www.linkedin.com/in/cole-soffa-656015107"),
  tags$br(),
  tags$em("https://www.kaggle.com/chicago/chicago-red-light-and-speed-camera-data")
)

server <- function(input, output, session) {
  
  output$plot <- renderggiraph({
    p = ggplot(monthly_violations_by_type) +
      geom_bar_interactive(aes(x = Month, y = Total_violations,
                               tooltip = Month_name, data_id = Month, fill = Type), stat = 'identity', position = 'stack', size = 2)+
      geom_hline_interactive(data = monthly_mean, aes(yintercept = m, group = Year, tooltip = round(m, digits = 0)), size = 1)+
      
      theme(panel.background = element_rect(fill = 'white'))+
      scale_x_discrete(name = 'Month', breaks = c('01', '03', '05', '07', '09', '11'), labels = c('01' = '1', '03' = '3', '05' = '5', '07' = '7', '09' = '9', '11' = '11'))+
      scale_y_continuous(labels = comma)+
      scale_fill_brewer(palette = 'Pastel1')+
      ylab("Total Violations/Month")+
      facet_wrap(~Year, nrow = 1)
    x <- ggiraph(ggobj = p , selection_type = 'single')
    x <- girafe_options(x, opts_hover(css = "fill:yellow;r:10pt;"))
  })
  
  output$text <- renderUI({
    if (is.null(input$plot_selected)) {
      paste('')
    }
    else {
      HTML("<h4>", "From an aggregate level, speed camera violations have consistently decreased since 2014. Is this proof that speed cameras have been effective in Chicago?", "</h4>")
    }
  })
  
  
  # This reactive function uses the selectInput UI to filter by type of speed camera (park or school)
  # Next, a monthly mean for each year is calculated which is shown as reference lines in the plot
  d <- reactive({
    df1 <- filter(monthly_violations_by_type, Type == input$type)
    
    year_mean <- df1 %>%
      group_by(Year, Type) %>%
      summarise(
        mean = mean(Total_violations)
      )
    
    if (input$type == 'Park') {
      fill = "#FBB4AE"
      month = "07"
    }
    else {
      fill = "#B3CDE3"
      month = "08"
    }
    values <- reactiveValues(d = df1, f = fill, m = year_mean, month = month)
  })
  
  output$plot1 <- renderggiraph({
    p1 <- ggplot(d()$d)+
      geom_point_interactive(aes(x = Month, y = Total_violations, tooltip = Month_name, data_id = Month), color = d()$f)+
      geom_point(data = filter(d()$d, Month == d()$month), aes(x = Month, y = Total_violations), size = 4.5, shape = 1)+
      geom_point(data = filter(d()$d, Month == d()$month), aes(x = Month, y = Total_violations), size = 4.5, shape = 1)+
      geom_line(aes(x = Month, y = Total_violations, group = Year), color = d()$f)+
      geom_hline_interactive(data = d()$m, aes(yintercept = mean, group = Year, tooltip = round(mean, digits = 0)), size = 1)+
      theme(panel.background = element_rect(fill = 'white'))+
      scale_x_discrete(name = 'Month', breaks = c('01', '03', '05', '07', '09', '11'), labels = c('01' = '1', '03' = '3', '05' = '5', '07' = '7', '09' = '9', '11' = '11'))+
      scale_y_continuous(labels = comma)+
      ylim(0, 100000)+
      ylab("Total Violations/Month")+
      facet_wrap(~Year, nrow = 1)
    
    y <- ggiraph(ggobj = p1, selection_type = 'single')
    y <- girafe_options(y, opts_hover(css = "fill:yellow;r:10pt;"))
    
  })
  
  # This output text serves as reactive annotation to the user clicking on the plot
  output$text1 <- renderUI({
    if (is.null(input$plot1_selected) == FALSE) { 
      if (input$type == "Park") {
        if (input$plot1_selected == "07") {
          HTML("<h4>", "Clearly, peak speed violations occur in the summer months (July and August) for park zones. One would hypothesize that their are more individuals driving near park areas in warmer temperatures. Likewise, the winter months (January and February) show the lowest number of speed violations throughout the year when temperatures are the coldest.", "</h4>")
        }
        else {
          paste("")
        }
      }
      else {
        if (input$plot1_selected == "08") {
          HTML("<h4>", "August has a drastically lower total number of speed violations each year within school zones. According to the city's policies, Chicago's school zone speed cameras are enforced only on weekdays at a 20 mph speed limit when children are present in the zone and are enforced at the posted speed limit when children are not present in the zone. Because the majority of schools are out and summer school is over, this makes sense why August has such a low number of speed violations.", "</h4>")
        }
        else {
          paste("") 
        }
      }
    }
    else {
      paste("")
    
    }
  })
  
  # Now the speed violations are grouped by each specific camera
  d3 <- reactive({
    # Total monthly violations each year for each camera
    df3 <- filter(modified_speed_violations, Type == input$type1) %>%
      group_by(Camera, Month, Year, LATITUDE, LONGITUDE, Type, Zone) %>%
      summarise(
        Total_violations = sum(VIOLATIONS)
      )
    
    # Average monthly violations for each year for each camera
    df3 <- ungroup(df3) %>%
      group_by(Camera, Year) %>%
      summarise(
        Average_violations = mean(Total_violations)
      )

    # Used for y limits in the plot
    max <- ungroup(df3) %>%
      summarise(
        m = max(Average_violations)
      )
    
    max <- max$m
    
    # The "top_camera" column consists of five or six cameras that I have identified as consistently having a high number of speed violations each year
    if (input$type1 == 'Park'){
      df3 <- mutate(df3, top_camera = ifelse(Camera %in% c('CHI149', 'CHI045', 'CHI021', 'CHI120', 'CHI003', 'CHI079'), 'Top', 'Regular'))
      fill <- "#FBB4AE"
    }
    
    else {
      df3 <- mutate(df3, top_camera = ifelse(Camera %in% c('CHI031', 'CHI028', 'CHI044', 'CHI146', 'CHI108'), 'Top', 'Regular'))
      fill = "#B3CDE3"
      
    }
    values <- reactiveValues(d = df3, f = fill, m = max, mean = mean)
    
  })
  
  output$plot2 <- renderggiraph({
    p2 <- ggplot(d3()$d, aes(x = Year, y = Average_violations))+
      geom_point_interactive(data = filter(d3()$d, Year == input$year, top_camera == 'Regular'), aes(tooltip = Camera, data_id = Camera), position = 'jitter', color = 'gray')+
      geom_point_interactive(data = filter(d3()$d, Year == input$year, top_camera == 'Top'), aes(tooltip = Camera, data_id = Camera), color = d3()$f, size = 3)+
      geom_path(data = filter(d3()$d, Year <= input$year, top_camera == 'Top'), aes(group = Camera), color = d3()$f, alpha = 0.35)+
      geom_smooth(data = filter(d3()$d, Year <= input$year, top_camera == 'Top'), color = d3()$f, se = FALSE)+
      geom_smooth(data = filter(d3()$d, Year <= input$year, top_camera == 'Regular'), color = 'gray', se = FALSE)+
      ylab('Avg violations/month')+
      theme(panel.background = element_rect(fill = 'white'),
            axis.title.x = element_blank())+
      ylab("Average violations/month")+
      xlim(2013, 2019)+
      ylim(0, d3()$m)
    
    z <- ggiraph(ggobj = p2, selection_type = 'multiple' )
    z <- girafe_options(z, opts_hover(css = "fill:yellow;r:10pt;"))
    
  })
  
  observeEvent(input$reset, {
    session$sendCustomMessage(type = 'plot2_set', message = character(0))
  })
  
  d4 <- reactive({
    df4 <- filter(modified_speed_violations, Camera == input$plot2_selected)
    
    if (input$type1 == 'Park'){
      fill <- "#FBB4AE"
    }
    else {
      fill = "#B3CDE3"
      
    }
    values <- reactiveValues(d = df4, f = fill)
  })
  
  output$text2 <- renderUI({
    if (input$year == 2018) { 
      HTML("<h4>", "In both park and school zones, there seem to be five or six speed cameras (highlighted in color) that consistently have a high volume of speed violations each year compared to the rest of the cameras (gray). It does seem, however, that these five or six cameras are converging closer to the average speed camera as time goes on.", "</h4>")
    }
    else {
      paste("")
      
    }
  })
  output$plot3 <- renderggiraph({
    p3 <- ggmap(map)+
      geom_point_interactive(data = d4()$d, aes(x = LONGITUDE, y = LATITUDE, tooltip = Zone), color = d4()$f, size = 3)+
      theme(axis.title = element_blank(),
            axis.ticks = element_blank(),
            axis.text = element_blank())
    
    c <- ggiraph(ggobj = p3, selection_type = 'single' )
    c <- girafe_options(c, opts_hover(css = "fill:yellow;r:10pt;"))
  })
  
  output$text3 <- renderUI({
    if (is.null(input$plot2_selected)) {
      paste('')
    }
    else {
      HTML("<b>", "Note: ", "</b>", "Hover over camera on map to show park or school zone it is located in")
    }
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)

