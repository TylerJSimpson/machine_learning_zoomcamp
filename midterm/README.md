# Midterm Project - Credit Risk Model 

## Description

This project uses a decision tree model for determining whether or not an applicant should be approved for a recreational vehicle loan.  
- Decision tree was chosen because of the nature of financial audits which require a clear breakdown of credit decisions.  
- The model's accuracy will likely be improved with the addition of more data. Particularly, credit report output data but it has been left off due to potential PII.  
- The model has been containerized as a docker image named `midterm`.


## Data Dictionary
Due to the sensitivity of the source data I did not do data extraction in Python in the [notebook.ipynb](notebook.ipynb). It was instead done using SQL on my internal database. Below I will define the fields and how NULLS were handled.

`is_joint` - bit field determining whether the application is jointly filed or not **NO NULLS**    
`applicant_credit_score` - credit score of the applicant **NULLS = -1**  
`coapplicant_credit_score` - credit score of the coapplicant **NULLS = -1**  
`is_boat`, `is_rv`, `is_engine` - one-hot-encoded bit field vehicle type which is the object for which the loan is originated **NO NULLS**    
`vehicle_age` - calculated on the fly from time of applicant vs year of make **NULLS = 0**  
`is_vehicle_new` - bit field for whether vehicle is new or old **NULLS = 0**  
`cash_price` - The cash price of the vehicle in USD **NO NULLS**  
`trade_in_value` - The cash price in USD of the collateral traded in on the loan **NULLS = 0**  
`trade_in_payoff` - The cash price in USD of what remains to be paid on the collateral if it is loaned on **NULLS = 0**   
`doc_fee` - The fees paid USD for the loan document processing **NULLS = 0**  
`amount_to_finance` - The total amount to finance on the loan in USD **NO NULLS**  
`payment` - The expected monthly payment in USD **NULLS = 0**  
`ltv` - The loan-to-value ie the amount owned vs the appraised value of an asset **NULLS = 0**  
`debt` - The amount of debt in USD the applicant currently holds **NULLS = 0**  
`down_payment` - The amount of down payment in USD that the applicant put down **NULLS = 0**  
`income` - The income reported by the applicant in USD **NULLS = 0**  
`proposed_dti` - The self-reported debt-to-income ratio of the applicant or coapplicant **NULLS = 0**  
`current_dti` - The calculated debt-to-income ratio of the applicant **NULLS = 0**  
`applicant_dti` - The self-reported debt-to-income ratio of the applicant **NULLS = 0**  
`applicant_age` - The calculated age of the applicant based on driver's license details and time of loan request **NULLS = -1**  
`is_funded` - Target value of 1 for funded and 0 for not funded **NO NULLS**  

## Virtual Environment

Pipenv was used as the python virtual environment tool for this project.  
All of the dependencies can be found in the [Pipfile](Pipfile)

Install pipenv:
```bash
pipenv install
```

Run pipenv:
```bash
pipenv shell
```

## Containerization

This project is containerized via a Docker image.  
The [Dockerfile](Dockerfile) specifies the python image base and installs pipenv and then installs the dependencies of the project via the [Pipfile](Pipfile) mentioned previously. It then grabs the prediction script [predict.py](predict.py) and model script [model.bin](model.bin) and specifies the entry point using flask.  

First you want to install docker which is dependent on your OS. Documentation can be found on the official docker [website](https://docs.docker.com/desktop/).  

Once docker is installed you want to first install the base docker python image:
```bash
docker run -it --rm --entrypoint=bash python:3.8.12-slim
```  

Now you can create the [Dockerfile](Dockerfile) specifying the dependencies and necessary files as well as expose the port for ocmmunication.

Using the Dockerfile we now build our image named `midterm`:
```bash
docker build -t midterm .
```

The original docker python image can be closed and we can run the new `midterm` image we built:
```bash
docker run -it --rm --entrypoint=bash midterm
```

Now open up the port in the docker image:
```bash
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

## Cloud deployment

First we want to push our image to Docker hub.  

If you haven't logged into docker yet:
```bash
docker login
```

To be able to push the local image to hub you first need to tag it to your login and desired repository name:
```bash
docker tag midterm tjsimpson/midterm:latest
```

Now it can be pushed to the hub:
```bash
docker push tjsimpson/midterm:latest
```

I chose to use Azure to deploy my container. This was done using their `Container App` service.  
The JSON template I used to deploy the container can be found [here](template.json).  
Alternatively the container app can be created in the portal all you need to specify is the name of the container in docker hub ie in my case `tjsimpson/midterm:latest`.  
