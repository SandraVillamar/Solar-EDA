[//]:<> (Titles)
<div align="center">
<h1>ECE143 Porject</h2>
<h3> Exploring Environmental Factors and their Impact on Solar Power
</h3>
</div>

<p style='text-align: right;'>Sandra Villamar</p>
<p style='text-align: right;'>Zhenduo Wen</p>
<p style='text-align: right;'>Zian Wang</p>
<p style='text-align: right;'> Shixuan Wu </p>

<p style='text-align: right;'>Zengrui Li</p>

&nbsp;
&nbsp;


<br />
    <a href="https://colab.research.google.com/drive/1lwJoR0XxA76lOJT2g6u6Y5HqkZaizeZT?usp=sharing"><strong>Explore the Colab docs »</strong></a>
     <br />
       <br />
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Problem">Problem</a>
</li>
    <li>
      <a href="#Dataset">Dataset</a>
      <ul>
        <li><a href="#Variables">Variabels</a></li>
   
</ul>
    </li>
    <li><a href="#Objective">Objective</a></li>
    <li><a href="#proposed-solution">Proposed Solution</a></li>
    
<li><a href="#real-world-application">Real World Application</a></li>
<li><a href="#Contents">Contents</a>

<ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      
</ul>
</li>
  </ol>
</details>
  <br />
       <br />
       
<!-- ABOUT THE PROJECT -->

## Problem
### How do various weather and location features relate to solar power output? Which features can be used to best predict solar power output in the absence of irradiance data? 
&nbsp;
&nbsp;

## Dataset
#### We will be using the following dataset: Williams, Jada; Wagner, Torrey (2019), “Northern Hemisphere Horizontal Photovoltaic Power Output Data for 12 Sites”, Mendeley Data, V5, doi: 10.17632/hfhwmn8w24.5 
#### This dataset contains power output from horizontal photovoltaic panels located at 12 Northern hemisphere sites over 14 months. It has 21K observations of 17 variables as follows: location, date, time sampled, latitude, longitude, altitude, year and month, month, hour, season, humidity, ambient temperature, power output from the solar panel, wind speed, visibility, pressure, and cloud ceiling. 
### Variables
1. Location:
    * Latitude
    * Longitude
2. Time 
    * Date
    * Time sampled
    * Season
3. Weather
    * Altitude
    * Humidity
    * Ambient temperature
    * Wind speed
    * Visibility
    * Pressure
    * Cloud Ceiling


&nbsp;
&nbsp;
## Objective
### Find the weather and location features that best explain solar power output
&nbsp;
&nbsp;
## Proposed Solution
 #### We plan to conduct an in-depth EDA in order to find the features that best explain power output. Then, we will implement four ML models and use these features as input to predict power output. We plan to build the following models and compare their performances: weighted linear regression, random forest, gradient boosting, and DNN. 

&nbsp;
&nbsp;

## Real World Application
#### Our solution can provide data-driven support on the cost revenue analysis for solar energy companies, especially those who cannot collect large-scale irradiance information in a short time. Solar energy is a rapidly growing market. In order to maintain solar plant sites and produce the required amount of energy, knowing solar power output is vital. This is normally modeled with irradiance data. But irradiance measurements require specific sensors, which take a long time to deploy and have a high cost. Hence, finding a way to predict solar power output with existing weather and location data would certainly be beneficial. 

&nbsp;
&nbsp;
## Contents:
	
1. Group Notebook
    * Packaage Needed:
        * chart_studio
        * pandas and numpy
        * plotly
        * seaborn
        * matplotlib

    ### Prerequisites

list of software and how to install them.
* pip
  ```sh
  pip install chart_studio
  ```
  ```sh
  pip install seaborn
  ```

2. Final Presentation Slides