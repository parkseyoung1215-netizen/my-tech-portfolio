"""New calibration100/heldout150:40 structural variants plus10 fact-only probes."""
import random
if __package__:
    from .solver300_dataset import prior_keys,task_key,ROOT
    from .verify_routing200 import reference_answer
else:
    from solver300_dataset import prior_keys,task_key,ROOT
    from verify_routing200 import reference_answer
SEED=812067

def build(data_dir):
    rng=random.Random(SEED);seen=prior_keys(data_dir);splits=[]
    facts=[
        ('Which planet has the greatest mass in the Solar System?','Jupiter',['Earth','Mars','Venus']),
        ('Which planet is nearest to the Sun?','Mercury',['Venus','Earth','Mars']),
        ('Which planet is commonly called the Red Planet?','Mars',['Venus','Neptune','Mercury']),
        ('Which planet is famous for its large visible ring system?','Saturn',['Earth','Mars','Venus']),
        ('Which star supplies most of the energy reaching Earth?','Sun',['Sirius','Polaris','Vega']),
        ('Which natural satellite orbits Earth?','Moon',['Phobos','Titan','Europa']),
        ('What mainly causes the regular cycle of day and night on Earth?','Earth rotating',['Earth changing size','The Sun orbiting Earth daily','Lunar eclipses']),
        ('What does the Moon mainly reflect to appear bright?','Sunlight',['Its own visible light','Radio waves','Light from Earth only']),
        ('During a solar eclipse, which body is between Earth and the Sun?','Moon',['Mars','Jupiter','Venus']),
        ('During a lunar eclipse, which body lies between the Sun and the Moon?','Earth',['Mars','Venus','Jupiter']),
        ('Which ocean covers the largest area?','Pacific',['Atlantic','Indian','Arctic']),
        ('Which continent contains the geographic South Pole?','Antarctica',['Australia','Africa','Europe']),
        ('Which river runs through Cairo?','Nile',['Amazon','Danube','Mekong']),
        ('Which mountain has the highest summit above sea level?','Everest',['Kilimanjaro','Fuji','Mont Blanc']),
        ('Which rock type forms when molten magma or lava cools?','Igneous',['Sedimentary','Metamorphic','Coal only']),
        ('Which rock type commonly forms by compaction of sediments?','Sedimentary',['Igneous','Metamorphic','Molten']),
        ('Which rock type is changed by heat and pressure without fully melting?','Metamorphic',['Igneous','Sedimentary','Liquid magma']),
        ('What is the usual chemical formula for water?','H2O',['CO2','O2','NaCl']),
        ('Which particle carries a positive elementary electric charge?','Proton',['Electron','Neutron','Photon']),
        ('Which particle carries a negative elementary electric charge?','Electron',['Proton','Neutron','Photon']),
        ('Which listed particle has no net electric charge?','Neutron',['Proton','Electron','Positron']),
        ('Which particles define the atomic number of an element?','Protons',['Neutrons only','Electrons plus neutrons','All nucleons']),
        ('What is counted by an atom\'s mass number?','Protons plus neutrons',['Electrons only','Protons minus neutrons','Photons']),
        ('Which force makes freely falling objects accelerate toward Earth?','Gravity',['Magnetism only','Friction only','Buoyancy only']),
        ('What is the name for energy associated with motion?','Kinetic energy',['Chemical symbol','Electric charge','Mass number']),
        ('What is light bouncing from a mirror called?','Reflection',['Refraction','Absorption only','Evaporation']),
        ('What is light changing direction when entering a different medium called?','Refraction',['Reflection only','Freezing','Condensation']),
        ('What separates white light into colors in a prism?','Dispersion',['Combustion','Fermentation','Sublimation']),
        ('How do opposite magnetic poles interact?','They attract',['They repel','They lose all magnetism','They turn into electric charges']),
        ('What does a barometer measure?','Atmospheric pressure',['Wind speed','Air humidity','Distance']),
        ('What does an anemometer measure?','Wind speed',['Pressure','Humidity','Electric current']),
        ('What does a hygrometer measure?','Humidity',['Temperature only','Force','Voltage']),
        ('What does a thermometer primarily measure?','Temperature',['Pressure','Mass','Current']),
        ('Which instrument records seismic ground motion?','Seismometer',['Barometer','Hygrometer','Ammeter']),
        ('Which optical instrument is designed to view very small objects?','Microscope',['Telescope','Barometer','Compass']),
        ('Which optical instrument is commonly used to observe distant stars?','Telescope',['Microscope','Thermometer','Ammeter']),
        ('What is the pivot point of a lever called?','Fulcrum',['Rotor','Lens','Valve']),
        ('Which simple machine is a sloped surface such as a ramp?','Inclined plane',['Pulley','Wheel and axle','Lever only']),
        ('Which simple machine uses a grooved wheel and rope?','Pulley',['Inclined plane','Wedge only','Screw only']),
        ('Which organelle is the main site of photosynthesis in plant cells?','Chloroplast',['Mitochondrion','Lysosome','Ribosome']),
        ('Which organelle produces much of the ATP in typical aerobic animal cells?','Mitochondrion',['Chloroplast','Cell wall','Vacuole']),
        ('Which covering is characteristic of birds?','Feathers',['Hair only','Moist bare skin','Scales only']),
        ('Which listed animal is an amphibian?','Frog',['Eagle','Salmon','Lizard']),
        ('Which listed adult animal obtains oxygen mainly through gills?','Salmon',['Dolphin','Whale','Seal']),
        ('Which group of animals has a backbone?','Vertebrates',['All insects','All worms','All jellyfish']),
        ('Which material forms the main structural component of plant cell walls?','Cellulose',['Glycogen','Hemoglobin','Cholesterol']),
        ('Which structure is present in plant cells but absent from animal cells?','Cell wall',['Cell membrane','Ribosome','DNA']),
        ('What is the outermost solid layer of Earth called?','Crust',['Outer core','Inner core','Mantle only']),
        ('Which layer lies immediately below Earth\'s crust?','Mantle',['Inner core','Atmosphere','Ocean only']),
        ('What is molten rock below Earth\'s surface called?','Magma',['Lava only','Sediment','Glacier']),
    ]
    rng.shuffle(facts);days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for split,n,offset in [('calibration',20,0),('fresh150',30,20)]:
        rows=[]
        def add(category,q,a,kind='choice',code='',regime='near'):
            key=task_key(q)
            if key in seen:raise ValueError('Guard split/old duplicate question')
            seen.add(key);rows.append({'question':q,'expected':str(a),'category':category,'task_format':kind,'reference_code':code,'regime':regime})
        def choice(category,q,a,wrong,code='',regime='near'):
            opts=[str(a)]+[str(v) for v in wrong]
            if len(opts)!=4 or len(set(opts))!=4:raise ValueError('Invalid options')
            rng.shuffle(opts);add(category,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts)), 'ABCD'[opts.index(str(a))],code=code,regime=regime)
        for i in range(n):
            shifted=split=='fresh150' and i>=20
            while True:
                a,b,c=rng.randint(3101,5700),rng.randint(17,71),rng.randint(211,401)
                q=f'({a} - {c}) * ({b} + 2)' if shifted else f'{a} - {c}' if i%2==0 else f'({a} * {b}) - {c}'
                if task_key(q) not in seen:break
            add('arithmetic',q,reference_answer(q),'integer',regime='shift' if shifted else 'near')
        for i,(q,a,w) in enumerate(facts[offset:offset+n]):choice('knowledge',q,a,w,regime='fact_holdout' if split=='fresh150' and i>=20 else 'near')
        for j in range(n//5):
            shifted=split=='fresh150' and j>=4;regime='shift' if shifted else 'near';k=j+(211 if split=='calibration' else 511)
            today=days[k%7];delta=k%17+18
            target=(k%7-delta)%7 if shifted else (k%7+delta)%7
            choice('logic',f'Reason about a weekly calendar: today is {today}. Which weekday was it {delta} days earlier?' if shifted else f'Reason about a weekly calendar: today is {today}. Which weekday is it {delta} days later?',days[target],rng.sample([d for d in days if d!=days[target]],3),regime=regime)
            names=[f'Aro{k}',f'Beni{k}',f'Cato{k}',f'Dina{k}'];rank=k%4+1
            rel=[f'{names[i+1]} finishes after {names[i]}' if shifted else f'{names[i]} finishes before {names[i+1]}' for i in range(3)];rng.shuffle(rel)
            choice('logic','Reason about a race: '+'; '.join(rel)+f'. Who finishes in position {rank}?',names[rank-1],[v for v in names if v!=names[rank-1]],regime=regime)
            x=[k+1,k+3,k+5];y=[k,k+3,k+6]
            if shifted:
                choice('logic',f'Reason about set membership: X={x}, Y={y}, Z={[k+3,k+8]}. Which value belongs to all three sets?',k+3,[k+1,k+5,k+8],regime=regime)
            else:choice('logic',f'Reason about set membership: X={x}, Y={y}. Which value belongs to both sets?',k+3,[k,k+1,k+6],regime=regime)
            bits=[rng.randrange(2) for _ in range(11)];ones=sum(bits)+(1 if shifted else 0)
            condition=' After appending one more 1 bit,' if shifted else ''
            choice('logic',f'Reason about bit parity case {k}: bits={bits}.{condition} Is the number of 1 bits odd or even?','Odd' if ones%2 else 'Even',['Even' if ones%2 else 'Odd','Cannot determine','Neither'],regime=regime)
            x,y,z,w=[f'N{k}{s}' for s in 'abcd']
            choice('logic',f'Reason about directed links: {x}->{y}, {y}->{z}, {z}->{w}. Starting at {x}, which node is reached after exactly '+('three' if shifted else 'two')+' links?',w if shifted else z,[v for v in (x,y,z,w) if v!=(w if shifted else z)],regime=regime)
            v=k%70+81
            codes=[(f'print(sum([n + 1 for n in range({v}, {v+7}, 2)]))' if shifted else f'print(sum(range({v}, {v+9}, 3)))',str(sum(n+1 for n in range(v,v+7,2))) if shifted else str(sum(range(v,v+9,3))),[str(v),str(v*3),'Error']),
                (f"s = 'red{k}\\tblue\\tgreen'\nprint(len(s.split('\\t')))" if shifted else f"s = '  leaf{k} red blue green  '\nprint(len(s.strip().split()))",'3' if shifted else '4',['2','5','Error']),
                (f'v = [{v}, {v+4}, {v+9}]\nprint(v[::-1][1])' if shifted else f'v = [{v}, {v+4}, {v+9}]\nprint(v[1:][0])',str(v+4),[str(v),str(v+9),'Error']),
                (f"d = {{'x': {v}, 'z': 4}}\nprint(d.get('z', 7) + d['x'])" if shifted else f"d = {{'x': {v}}}\nprint(d.get('z', 9) + d['x'])",str(v+4) if shifted else str(v+9),[str(v),'9','Error']),
                (f'print([n+1 for n in range({v}, {v+6}) if n % 2 != 0])' if shifted else f'print([n for n in range({v}, {v+6}) if n % 2 == 1])',str([n+1 for n in range(v,v+6) if n%2]) if shifted else str([n for n in range(v,v+6) if n%2]),[str(list(range(v,v+6))),'[]','Error'])]
            for code,a,wrong in codes:choice('coding','What does this Python program output?\n```python\n'+code+'\n```',a,wrong,code,regime)
            person=f'Nora{k}';other=f'Olin{k}'
            choice('comprehension',f'Read the passage: Reservation {k} changed from Room Oak to Room Cedar.'+(' It was then changed back to Room Oak.' if shifted else '')+' What room is finally reserved?','Room Oak' if shifted else 'Room Cedar',['Room Cedar' if shifted else 'Room Oak','Both rooms','No room'],regime=regime)
            amount=k%80+31;refund=k%13+11
            choice('comprehension',f'Read the passage: Receipt {k} records payment {amount} coins and refund {refund} coins.'+(' A separate fee is 3 coins and is not part of the refund.' if shifted else '')+' What is the refund amount?',refund,[amount,amount-refund,'Zero'],regime=regime)
            choice('comprehension',f'Read the passage: {person} composed a song. {other} performed it.'+(' A second song was composed by someone else, not mentioned by name.' if shifted else '')+' Who composed the first song?',person,[other,'Both named people','Nobody'],regime=regime)
            choice('comprehension',f'Read the passage: Notice {k} says all parks close at 17:00 except Park Fern, which closes at 19:00.'+(' On Sundays Fern closes at 16:00. Today is Sunday.' if shifted else '')+' When does Park Fern close today?','16:00' if shifted else '19:00',['17:00','19:00' if shifted else '16:00','Never'],regime=regime)
            choice('comprehension',f'Read the passage: Order {k} went from queued to shipped to returned.'+(' It was then shipped again.' if shifted else '')+' What is its latest state?','Shipped' if shifted else 'Returned',['Returned' if shifted else 'Shipped','Queued','Unknown'],regime=regime)
        if len(rows)!=n*5:raise ValueError('Unbalanced tasks')
        rng.shuffle(rows);splits.append(rows)
    return tuple(splits)
