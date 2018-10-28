import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

############# These are pre-written helper functions ##########################
####### Please go to the empty main() function below to start #################

# Generate some data simulating the home schedule of an individual
def getData(days):
    weekday = [0.8, 1, 1, 1, 1, 1, 0.8, 0.7, 0.5, 0.1, 0, 0, # <- 11-12
         0, 0, 0, 0.1, 0.3, 0.8, 0.8, 0.9, 0.9, 1, 1, 1]
    saturday = [0, 0, 0, 0.5, 1, 1, 1, 1, 1, 1, 1, 0.9,
         0.9, 0.8, 0.6, 0.5, 0.2, 0.2, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3]
    sunday = [0.1, 0.1, 0.3, 0.9, 1, 1, 1, 1, 1, 1, 0.8, 0.8,
         1, 1, 0.8, 0.8, 0.8, 0.3, 0.2, 0.1, 0.9, 1, 1, 1]
    schedule = [
        weekday,
        weekday,
        weekday,
        weekday,
        weekday,
        saturday,
        sunday
    ]

    # for ease of training ML models, output the data as the independent variables,
    # [day, hour] and the label, which is 1 or 0 if the person is at home or not
    variables = []
    labels = []
    day = 0
    while day < days:
        hourOfDay = 0
        dayOfWeek = day % 7
        daySchedule = schedule[dayOfWeek]
        for hourProbability in daySchedule:
            variables.append([dayOfWeek, hourOfDay])
            hourOfDay += 1
            labels.append(1 if random.randint(0,100)/100 < hourProbability else 0)
        day += 1

    return { 'variables': variables, 'labels': labels }

# return an example of a particular day of the week
def getDay(day):
    days = getData(day+1)
    startHour = 24 * day
    return {
        'variables': days['variables'][startHour:startHour+24],
        'labels': days['labels'][startHour:startHour+24]
    }

# supports plotting multiple days as a bar chart
def plotData(labels, filename):
    # Assumes full days
    numberOfDays = int(len(labels)/24)
    colwidth = 0.6/numberOfDays

    ax = plt.subplot(111)
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for day in range(0, numberOfDays):
        basex = range(0,24)
        x = []
        for i in basex:
            x.append(i-0.5+day*colwidth)

        startHour = day*24
        y = labels[startHour:startHour+24]
        ax.bar(x, y, width=colwidth, color=colours[day%7], align='center')
    fig = plt.gcf()
    fig.savefig(filename if filename else str(numberOfDays)+"_days.png")
    plt.clf()

# test the model
def testModel(model, day, filename):
    predicted = model.predict(day['variables']).tolist()
    plotData(predicted + day['labels'], filename)

################### Now comes the fun part, keep scrolling ####################

















######################### Start here! #########################################
# this is where it all comes together
def main():
    # Examine the data
    day = getData(1)
    y = day['labels']
    plotData(y, "monday.png")

    monTue = getData(2)['labels']
    plotData(monTue, "mon_tue.png")
    friSat = getData(7)['labels'][96:144]
    plotData(friSat, "fri_sat.png")

    # Train the model
    days = getData(5) # 50
    model = KNeighborsClassifier()
    model.fit(days['variables'], days['labels'])

    # Test
    testModel(model, getDay(0), "monday-predict.png")
    testModel(model, getDay(5), "saturday-predict.png")


main() # run the program
