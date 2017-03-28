#!/usr/bin/python
import sys
import csv

n = 0
n_feature = 0

for line in csv.reader(sys.stdin, delimiter = ','):
    if len(line) == 95:
        id = line[0]
        name = line[4]
        summary = line[5]
        space = line[6]
        description = line[7]
        neighborhood_overview = line[9]
        notes = line[10]
        transit = line[11]
        access = line[12]
        interaction = line[13]
        house_rules = line[14]

        neighborhood = line[39]
        city = line[41]
        state = line[42]
        zipcode = line[43]

        latitude = line[48]
        longitude = line[49]

        property_type = line[51]
        room_type = line[52]

        accommodates = line[53]
        bathrooms = line[54]
        bedrooms = line[55]
        beds = line[56]
        bed_type = line[57]

        amenities = line[58]
        square_feet = line[59]

        price = line[60]


        feature = description

        n = n + 1
        if feature.strip() != '':
            n_feature = n_feature + 1




        #output = '{0}'.format(square_feet)
        #print(output)

print('total number of entries: {0}'.format(n))
print('entries with description: {0}'.format(n_feature))
print('percentage: {0}%'.format(float(n_feature) / n * 100))
