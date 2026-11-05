"""Fresh disjoint calibration100/evaluation150; labels never enter runtime."""
import random
if __package__:
    from .solver300_dataset import prior_keys,task_key,ROOT
    from .verify_routing200 import reference_answer
else:
    from solver300_dataset import prior_keys,task_key,ROOT
    from verify_routing200 import reference_answer
SEED=731942

def build(data_dir):
    rng=random.Random(SEED);seen=prior_keys(data_dir);splits=[]
    facts=[
        ('Which chemical element has the symbol He?','Helium',['Hydrogen','Hafnium','Holmium']),
        ('Which chemical element has the symbol Li?','Lithium',['Lead','Lanthanum','Lutetium']),
        ('Which chemical element has the symbol Be?','Beryllium',['Boron','Barium','Bromine']),
        ('Which chemical element has the symbol B?','Boron',['Bromine','Barium','Beryllium']),
        ('Which chemical element has the symbol C?','Carbon',['Calcium','Chlorine','Cobalt']),
        ('Which chemical element has the symbol N?','Nitrogen',['Neon','Nickel','Sodium']),
        ('Which chemical element has the symbol O?','Oxygen',['Osmium','Gold','Silver']),
        ('Which chemical element has the symbol F?','Fluorine',['Iron','Francium','Fermium']),
        ('Which chemical element has the symbol Ne?','Neon',['Nitrogen','Nickel','Sodium']),
        ('Which chemical element has the symbol Na?','Sodium',['Nitrogen','Neon','Nickel']),
        ('Which chemical element has the symbol Mg?','Magnesium',['Manganese','Mercury','Molybdenum']),
        ('Which chemical element has the symbol Al?','Aluminium',['Argon','Silver','Gold']),
        ('Which chemical element has the symbol Si?','Silicon',['Silver','Sulfur','Sodium']),
        ('Which chemical element has the symbol P?','Phosphorus',['Potassium','Platinum','Lead']),
        ('Which chemical element has the symbol S?','Sulfur',['Silicon','Silver','Sodium']),
        ('Which chemical element has the symbol Cl?','Chlorine',['Carbon','Calcium','Cobalt']),
        ('Which chemical element has the symbol Ar?','Argon',['Silver','Gold','Aluminium']),
        ('Which chemical element has the symbol K?','Potassium',['Krypton','Calcium','Carbon']),
        ('Which chemical element has the symbol Ca?','Calcium',['Carbon','Cadmium','Chlorine']),
        ('Which chemical element has the symbol Fe?','Iron',['Fluorine','Francium','Fermium']),
        ('Which chemical element has the symbol Cu?','Copper',['Carbon','Calcium','Cobalt']),
        ('Which chemical element has the symbol Zn?','Zinc',['Zirconium','Xenon','Nitrogen']),
        ('Which chemical element has the symbol Ag?','Silver',['Gold','Argon','Aluminium']),
        ('Which chemical element has the symbol Au?','Gold',['Silver','Argon','Aluminium']),
        ('Which chemical element has the symbol Pb?','Lead',['Phosphorus','Platinum','Potassium']),
        ('Which organ pumps blood through the human circulatory system?','Heart',['Liver','Lung','Stomach']),
        ('Which organ primarily filters blood to produce urine?','Kidney',['Heart','Stomach','Pancreas']),
        ('Which organ produces bile?','Liver',['Heart','Lung','Spleen']),
        ('Which organ stores bile before release?','Gallbladder',['Kidney','Spleen','Bladder']),
        ('Which organ contains the vocal cords?','Larynx',['Esophagus','Stomach','Liver']),
        ('Which plant structure primarily absorbs water from soil?','Roots',['Flowers','Fruit','Seeds']),
        ('Which green pigment is central to plant photosynthesis?','Chlorophyll',['Hemoglobin','Melanin','Keratin']),
        ('Which gas do plants consume during photosynthesis?','Carbon dioxide',['Helium','Hydrogen','Neon']),
        ('Which gas is released by oxygenic photosynthesis?','Oxygen',['Nitrogen','Helium','Methane']),
        ('Which molecules carry most hereditary information in human cells?','DNA',['Starch','Cellulose','Cholesterol']),
        ('Which blood cells mainly carry oxygen?','Red blood cells',['Platelets','Neurons','Bone cells']),
        ('Which blood components help initiate clotting?','Platelets',['Neurons','Egg cells','Muscle cells']),
        ('Which structure encloses most DNA in a typical human cell?','Nucleus',['Cell wall','Vacuole','Flagellum']),
        ('Which cellular structure is the main site of protein synthesis?','Ribosome',['Lysosome','Cell wall','Vacuole']),
        ('Which cell structure controls much of the exchange with its surroundings?','Cell membrane',['Chromosome','Nucleolus','Ribosome']),
        ('Which tissue connects muscle to bone?','Tendon',['Ligament','Cartilage','Epidermis']),
        ('Which tissue connects bone to bone at joints?','Ligament',['Tendon','Neuron','Epidermis']),
        ('Which process converts liquid water to water vapor?','Evaporation',['Freezing','Condensation','Melting']),
        ('Which process converts water vapor to liquid droplets?','Condensation',['Evaporation','Melting','Sublimation']),
        ('Which SI unit measures electric current?','Ampere',['Volt','Ohm','Watt']),
        ('Which SI unit measures electric resistance?','Ohm',['Ampere','Volt','Joule']),
        ('Which SI unit measures electric potential difference?','Volt',['Ampere','Ohm','Hertz']),
        ('Which SI unit measures frequency?','Hertz',['Newton','Pascal','Joule']),
        ('Which SI unit measures force?','Newton',['Pascal','Watt','Kelvin']),
        ('Which SI unit measures pressure?','Pascal',['Newton','Joule','Ampere']),
    ]
    rng.shuffle(facts)
    for split,n,start in [('calibration',20,0),('evaluation',30,20)]:
        rows=[]
        def add(category,q,answer,kind='choice',code=None):
            key=task_key(q)
            if key in seen:raise ValueError('Old/overlapping RCAV task stem')
            seen.add(key);row={'question':q,'category':category,'expected':str(answer),'task_format':kind,'reference_code':code or ''}
            rows.append(row)
        def choice(category,q,answer,wrong,code=None):
            opts=[str(answer)]+[str(v) for v in wrong]
            if len(opts)!=4 or len(set(opts))!=4:raise ValueError('Invalid options')
            rng.shuffle(opts)
            add(category,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts)),'ABCD'[opts.index(str(answer))],code=code)
        for i in range(n):
            while True:
                a,b,c=rng.randint(1201,2900),rng.randint(13,57),rng.randint(101,299)
                q=f'{a} - {c}' if i%3==0 else f'{a} * {b}' if i%3==1 else f'({a} - {c}) * {b}'
                if task_key(q) not in seen:break
            add('arithmetic',q,reference_answer(q),'integer')
        for q,a,wrong in facts[start:start+n]:choice('knowledge',q,a,wrong)
        days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        for j in range(n//5):
            number=j+(11 if split=='calibration' else 51)
            today=days[number%7];offset=number+3;answer=days[(number%7+offset)%7]
            choice('logic',f'Reason about a weekly calendar: today is {today}. Which weekday is {offset} days later?',answer,rng.sample([d for d in days if d!=answer],3))
            names=[f'Ivo{number}',f'Juna{number}',f'Karo{number}',f'Lumi{number}'];rank=(number%4)+1
            rel=[f'{names[k]} finishes before {names[k+1]}' for k in range(3)];rng.shuffle(rel)
            choice('logic',f'Reason about a four-person race: '+ '; '.join(rel)+f'. Who finishes in position {rank}?',names[rank-1],[v for v in names if v!=names[rank-1]])
            values=[number+2,number+4,number+6];others=[number+1,number+4,number+7]
            choice('logic',f'Reason about set membership: X={values} and Y={others}. Which value belongs to both?',number+4,[number+1,number+2,number+7])
            bits=[rng.randrange(2) for _ in range(9)];bits[0]=number%2
            choice('logic',f'Reason about bit parity case {number}: bits={bits}. Is the count of 1 bits odd or even?', 'Odd' if sum(bits)%2 else 'Even',['Even' if sum(bits)%2 else 'Odd','Neither','Cannot determine'])
            x,y,z,w=[f'P{number}{c}' for c in 'abcd']
            choice('logic',f'Reason about directed links: {x}->{y}, {z}->{y}, {y}->{w}. Starting at {x}, which listed node can be reached after exactly two links?',w,[x,y,z])
            k=number+10
            codes=[(f'v = [{k}, {k+2}, {k+5}]\nprint(v[1:][0])',str(k+2),[str(k),str(k+5),'Error']),
                (f"d = {{'x': {k}}}\nprint(d.get('z', 7) + d['x'])",str(k+7),[str(k),'7','Error']),
                (f'print([n for n in range({k}, {k+5}) if n % 2 == 0])',str([v for v in range(k,k+5) if v%2==0]),[str(list(range(k,k+5))),'[]','Error']),
                (f"s = '  wave{number} blue sky  '\nprint(len(s.strip().split()))",'3',['2','4','Error']),
                (f'print(sum(range({k}, {k+7}, 2)))',str(sum(range(k,k+7,2))),[str(k+7),str(k*4),'Error'])]
            for code,a,wrong in codes:choice('coding','What does this Python program output?\n```python\n'+code+'\n```',a,wrong,code)
            person,second=f'Mira{number}',f'Niko{number}'
            choice('comprehension',f'Read the passage: Booking {number} originally used Room Elm. A correction replaced Elm with Room Pine. Which room is now booked?','Room Pine',['Room Elm','Both rooms','No room'])
            choice('comprehension',f'Read the passage: {person} paid {k} coins and received a refund of {number} coins. The receipt explicitly lists the refund as {number} coins. How many coins were refunded?',str(number),[str(k),str(k-number),'Zero'])
            choice('comprehension',f'Read the passage: {person} wrote a poem. {second} copied that poem without composing a new one. Who composed the poem?',person,[second,'Both composed it','Nobody'])
            choice('comprehension',f'Read the passage: On schedule {number}, all shops close at 18:00 except Shop Iris, which closes at 20:00. At what time does Shop Iris close?','20:00',['18:00','19:00','Never'])
            choice('comprehension',f'Read the passage: Ticket {number} began as pending, then became confirmed, then was cancelled. No later change occurred. What is its final status?','Cancelled',['Pending','Confirmed','Unknown'])
        if len(rows)!=5*n:raise ValueError('Unbalanced split')
        rng.shuffle(rows);splits.append(rows)
    return tuple(splits)
