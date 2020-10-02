#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import Counter
from collections import defaultdict
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings


# In[ ]:


le = LabelEncoder()
le.fit(['Protoss','Terran','Zerg'])


# In[ ]:


commands = ['Attack','EvolveFlyerCarapace1','BuildRoboticsBay','EvolveGlialReconstitution','TrainSCV','UpgradeTerranInfantryWeapons1','BuildGhostAcademy','RaiseSupplyDepot','CancelCreepTumor','Feedback','BuildFusionCore','BuildReactorStarport','BuildBanelingNest','PhasingMode','UpgradeToLair','UnburrowWidowMine','Cheer','UnburrowHydralisk','BuildArmory','BuildReactorFactory','UpgradeTerranInfantryWeapons3','ResearchAnionPulseCrystals','HallucinateVoidRay','CancelBarracksAddon','EvolvePneumatizedCarapace','ResearchGraviticBoosters','UnloadTargetMedivac','ResearchTransformationServos','ChronoBoost','ResearchCharge','LiftFactory','ResearchPersonalCloaking','LiftBarracks','Charge','ForceField','Consume','BuildNuke','UpgradeGroundArmor3','UnburrowDrone','ResearchWeaponRefit','BuildNexus','SalvageShared','HallucinateStalker','MorphMutalisk','Abduct','HallucinatePhoenix','BuildSpineCrawler','EvolveGroundCarapace1','EvolveFlyerAttacks1','BurrowHydralisk','SiegeMode','TrainTempest','Blink','UnburrowBaneling','TrainStalker','UnburrowSwarmHost','TankMode','ResearchBlink','SeekerMissile','HaltBuilding','BuildTechLabFactory','UnloadAllBunker','YamatoGun','LoadTargetBunker','UnburrowInfestor','TrainSentry','WarpInDarkTemplar','ArmorpiercingMissiles','ArchonWarpSelection','LandFactory','ResearchBehemothReactor','TemporalField','CancelMorph','LiftStarport','UpgradeGroundWeapons2','BuildNydusNetwork','MorphToBaneling','TrainColossus','EvolveChitinousPlating','CancelGravitonBeam','CreepTumor','UpgradeTerranInfantryArmor3','BuildRefinery','BuildNydusWorm','UpgradeAirWeapons2','BurrowSwarmHost','TrainRaven','LowerSupplyDepot','MorphDrone','Envision','TrainReaper','ExtraSupplies','BurrowInfestor','UpgradeAirArmor1','TrainBanshee','FungalGrowth','EvolveFlyerCarapace3','HallucinateZealot','SpawnLarva','BuildSpawningPool','LiftCommandCenter','BuildHydraliskDen','BuildDarkShrine','UpgradeAirArmor3','TrainZealot','EvolveAdrenalGlands','ScannerSweep','VehicleAndShipPlating1','EvolveCentrifugalHooks','SpawnLocusts','BattleMode','ResearchConcussiveShells','FighterMode','UnloadAllNydus','EvolveTunnelingClaws','UpgradeGroundArmor2','TacticalNukeStrike','MorphRoach','TrainObserver','UnburrowQueen','BuildAutoTurret','CancelMorphToOverseer','ResearchExtendedThermalLance','GenerateCreep','MorphOverlord','TransportMode','VehicleAndShipPlating3','EvolveNeuralParasite','LandBarracks','BuildGateway','BuildExtractor','MassRecallMothershipCore','BuildCreepTumor','DecloakGhost','ScanMove','ResearchCaduceusReactor','HallucinateArchon','Contaminate','TrainProbe','UpgradeToMothership','BuildSpire','ResearchPsiStorm','StopRedirect','CancelStarportAddon','PsionicStorm','CloakBanshee','AssaultMode','BurrowBaneling','HallucinateOracle','SCVRepair','EvolveMeleeAttacks1','StrikeMode','TransformToWarpGate','HallucinateImmortal','UpgradeAirWeapons3','TrainHighTemplar','EvolveMissileAttacks2','EvolveMuscularAugments','UpgradeShields2','LandStarport','BuildTechLabStarport','UpgradeVehicleWeapons2','EvolveMeleeAttacks3','GravitonBeam','InfestorNeuralParasite','ResearchGraviticDrive','MorphSwarmHost','UpgradesShields3','MorphZergling','ResearchMoebiusReactor','EvolveBurrow','MothershipCorePurifyNexus','UpgradeVehicleWeapons1','LandOrbitalCommand','SetRallyPoint','BuildPhotonCannon','UprootSporeCrawler','CalldownMULE','EvolveMeleeAttacks2','UpgradeToOrbitalCommand','TrainViking','UnloadTargetWarpPrism','UnburrowZergling','SetUnitRally','UpgradeGroundArmor1','HallucinateProbe','BuildHatchery','BuildForge','BuildBarracks','EvolveGroovedSpines','MothershipMassRecall','StopGenerateCreep','BuildFactory','BlindingCloud','CancelLast','UpgradeStructureArmor','CancelRootSpineCrawler','CancelTacticalNukeStrike','ResearchStimpack','BuildInfestationPit','Stop','TrainGhost','DecloakBanshee','BuildStarport','SetWorkerRally','MorphHydralisk','GuardianShield','EvolveFlyerAttacks2','SpawnInfestedTerran','BuildBattleHellion','UseStimpack','Gather','OracleWeapon','UpgradeShipWeapons1','BuildAssimilator','OracleWeaponOff','MULERepair','BurrowUltralisk','Corruption','BuildPylon','UnloadTargetOverlord','BuildSporeCrawler','EvolveMetabolicBoost','TrainImmortal','TrainMedivac','BuildWidowMine','BurrowQueen','ReactorBarracks','BuildSensorTower','BurrowZergling','ResearchNeosteelFrame','HallucinateColossus','SniperRound','StimpackRedirect','Revelation','CAbil','TrainMothershipCore','ResearchCloakingField','TrainMarauder','BuildSiegeTank','UnburrowRoach','UpgradeGroundWeapons3','UpgradeTerranInfantryArmor2','BuildTwilightCouncil','MorphToOverseer','CancelMorphToGreaterSpire','BuildHellion','TrainWarpPrism','Move','RootSpineCrawler','UpgradeAirArmor2','BuildBunker','QueenTransfusion','EvolveGroundCarapace2','BuildMissileTurret','EvolveEnduringLocusts','UpgradeTerranInfantryWeapons2','BuildInterceptor','EvolvePathogenGlands','BuildFleetBeacon','TrainVoidRay','UnloadAllCommandCenter','ResearchFluxVanes','LiftOrbitalCommand','ResearchDrillingClaws','UpgradeAirWeapons1','LandCommandCenter','EvolveGroundCarapace3','ResearchCombatShield','EvolveVentralSacs','BuildStargate','VehicleAndShipPlating2','CancelUpgradeToHive','ResearchInfernalPreIgniter','TransformToGateway','UpgradeToHive','EvolveMissileAttacks3','TrainBattlecruiser','HoldPosition','LoadAllCommandCenter','UnburrowUltralisk','WarpInHighTemplar','BurrowDrone','TrainQueen','BurrowRoach','MorphToBroodLord','MorphUltralisk','BuildCommandCenter','UpgradeShields1','BuildEvolutionChamber','RootSporeCrawler','CancelUpgradeToOrbitalCommand','BuildThor','MorphInfestor','CancelTerranBuilding','CancelFactoryAddOn','BuildSupplyDepot','EMPRound','WarpInStalker','CancelUpgradeToLair','UprootSpineCrawler','BuildEngineeringBay','PrismaticAlignment','ResearchHiSecAutoTracking','ResearchWarpGate','TrainMarine','Patrol','MorphToGreaterSpire','TrainCarrier','CloakGhost','EvolveFlyerCarapace2','TrainOracle','BuildUltraliskCavern','UpgradeGroundWeapons1','UpgradeToPlanetaryFortress','TrainPhoenix','Dance','BurrowWidowMine','DisableVolatileBurst','MorphCorruptor','EvolveFlyerAttacks3','SpawnChangeling','MedivacSpeedBoost','WarpInZealot','EvolveMissileAttacks1','ResearchDurableMaterials','TechLabBarracks','Explode','BuildRoachWarren','UpgradeTerranInfantryArmor1','HallucinateWarpPrism','BuildRoboticsFacility','BuildTemplarArchive','WarpInSentry','UpgradeVehicleWeapons3','CancelBuilding','MorphViper','ExplosiveMissiles','ReturnCargo','HoldFireGhost','ResearchCorvidReactor','BuildPointDefenseDrone','CancelUpgradeToPlanetaryFortress','BuildCyberneticsCore']
selections = ['SiegeTank', 'UltraliskBurrowed', 'Corruptor', 'BanelingCocoon', 'Spire', 'ChangelingZealot', 'Lyote', 'Observer', 'CreepTumorBurrowed', 'MothershipCore', 'SupplyDepot', 'InfestationPit', 'WarpPrism', 'HydraliskDen', 'SporeCrawler', 'Ghost', 'MineralField', 'Viper', 'BarracksReactor', 'ChangelingMarine', 'UrsadakCalf', 'Zergling', 'Armory', 'FactoryFlying', 'Drone', 'BarracksFlying', 'EngineeringBay', 'StarportFlying', 'RoachBurrowed', 'UnbuildableRocksDestructible', 'SpineCrawlerUprooted', 'Carrier', 'RoachWarren', 'Roach', 'ChangelingMarineShield', 'DestructibleRockEx1DiagonalHugeULBR', 'Extractor', 'XelNagaTower', 'SCV', 'LabBot', 'ChangelingZergling', 'KarakFemale', 'FleetBeacon', 'Bunker', 'FactoryReactor', 'DestructibleRampDiagonalHugeBLUR', 'FactoryTechLab', 'Queen', 'CommandCenterFlying', 'PhotonCannon', 'Hellion', 'VoidRay', 'HighTemplar', 'BanelingBurrowed', 'RoboticsBay', 'Colossus', 'BanelingNest', 'InfestorBurrowed', 'Stargate', 'LabMineralField', 'Assimilator', 'DestructibleDebris6x6', 'SupplyDepotLowered', 'TechLab', 'AutoTurret', 'Changeling', 'OverseerCocoon', 'Tempest', 'SpawningPool', 'UnbuildableBricksDestructible', 'MissileTurret', 'Pylon', 'Phoenix', 'Infestor', 'OrbitalCommandFlying', 'GhostAcademy', 'WarpGate', 'Broodling', 'CyberneticsCore', 'Banshee', 'CreepTumorQueen', 'SporeCrawlerUprooted', 'CollapsibleRockTowerDiagonal', 'Battlecruiser', 'ProtossVespeneGeyser', 'UnbuildablePlatesDestructible', 'Reactor', 'Medivac', 'QueenBurrowed', 'Ultralisk', 'DroneBurrowed', 'BroodLordCocoon', 'InfestedTerran', 'SwarmHostBurrowed', 'Marauder', 'SensorTower', 'DarkShrine', 'Mutalisk', 'Immortal', 'BattleHellion', 'Hive', 'SpineCrawler', 'Archon', 'GreaterSpire', 'Scantipede', 'Nexus', 'Factory', 'UltraliskCavern', 'Larva', 'BarracksTechLab', 'RoboticsFacility', 'MULE', 'DestructibleIce6x6', 'WidowMineBurrowed', 'Refinery', 'Locust', 'DestructibleDebrisRampDiagonalHugeULBR', 'Marine', 'StarportTechLab', 'InfestedTerransEgg', 'NydusWorm', 'CollapsibleRockTowerRampRight', 'Forge', 'PointDefenseDrone', 'Probe', 'Reaper', 'Viking', 'SiegeTankSieged', 'Stalker', 'DestructibleRock6x6', 'Raven', 'DestructibleRockEx16x6', 'HydraliskBurrowed', 'FusionCore', 'ZerglingBurrowed', 'Overlord', 'Thor', 'Starport', 'CollapsibleRockTowerDebrisRampRight', 'Hatchery', 'Oracle', 'SwarmHost', 'Gateway', 'Sentry', 'OrbitalCommand', 'Hydralisk', 'RichMineralField', 'Barracks', 'Overseer', 'BroodLord', 'Egg', 'SpacePlatformGeyser', 'CollapsibleRockTowerDebris', 'StarportReactor', 'VikingAssault', 'Baneling', 'DarkTemplar', 'PlanetaryFortress', 'WidowMine', 'NydusNetwork', 'TwilightCouncil', 'CreepTumor', 'ChangelingZerglingWings', 'Lair', 'TemplarArchive', 'CommandCenter', 'Mothership', 'VespeneGeyser', 'DestructibleRockEx1DiagonalHugeBLUR', 'EvolutionChamber', 'Zealot', 'WarpPrismPhasing']
htkeys = ['ht_00', 'ht_01', 'ht_02', 'ht_10', 'ht_11', 'ht_12', 'ht_20', 'ht_21', 'ht_22', 'ht_30', 'ht_31', 'ht_32', 'ht_40', 'ht_41', 'ht_42', 'ht_50', 'ht_51', 'ht_52', 'ht_60', 'ht_61', 'ht_62', 'ht_70', 'ht_71', 'ht_72', 'ht_80', 'ht_81', 'ht_82', 'ht_90', 'ht_91', 'ht_92']
buildings = ['BuildTemplarArchive','BuildNexus','BuildHatchery','BuildSpineCrawler','BuildSpire','BuildStargate','BuildCyberneticsCore','BuildPointDefenseDrone','BuildTechLabFactory','BuildNydusWorm','BuildCreepTumor','BuildRoboticsFacility','BuildRefinery','BuildFusionCore','BuildWidowMine','BuildSpawningPool','BuildBanelingNest','BuildNuke','BuildCommandCenter','BuildEngineeringBay','BuildSporeCrawler','BuildHydraliskDen','BuildBattleHellion','BuildBarracks','BuildExtractor','BuildFactory','BuildGateway','BuildStarport','BuildPylon','BuildInfestationPit','BuildInterceptor','BuildDarkShrine','BuildReactorFactory','BuildFleetBeacon','BuildTechLabStarport','BuildSensorTower','BuildUltraliskCavern','BuildHellion','BuildThor','BuildRoachWarren','BuildBunker','BuildTwilightCouncil','BuildRoboticsBay','BuildAssimilator','BuildSupplyDepot','BuildMissileTurret','BuildSiegeTank','BuildPhotonCannon','BuildArmory','BuildNydusNetwork','BuildReactorStarport','BuildForge','BuildAutoTurret','BuildGhostAcademy','BuildEvolutionChamber']


# In[ ]:


def player_and_race(game):
    return {'player' : ':'.join(game[0]), 'race' : game[1][0]}


# In[ ]:


def cmd_nb_use(game):
    result = Counter()
    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'BasicCommandEvent':
                cmd = l[2]
            elif l[1] == 'TargetPointCommandEvent':
                cmd = l[-1]
            elif l[1] == 'TargetUnitCommandEvent':
                cmd = l[2]
            cmd = cmd.replace('\n','')
            if cmd != '':
                result.update([cmd + '_nb'])
    total = sum(result.values())
    result = {k:result[k]/total * 10000 for k in result}
    return result


# In[ ]:


def first_cmd_use(game):
    result = {}
    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'BasicCommandEvent':
                cmd = l[2]
            elif l[1] == 'TargetPointCommandEvent':
                cmd = l[-1]
            elif l[1] == 'TargetUnitCommandEvent':
                cmd = l[2]
            cmd = cmd.replace('\n','')
            if cmd != '' and (cmd + '_ft') not in result:
                result[cmd + '_ft'] = l[0]
    return result


# In[ ]:


def first_cmd_uses(game, nb):
    tempo = defaultdict(list)
    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'BasicCommandEvent':
                cmd = l[2]
            elif l[1] == 'TargetPointCommandEvent':
                cmd = l[-1]
            elif l[1] == 'TargetUnitCommandEvent':
                cmd = l[2]
            cmd = cmd.replace('\n','')
            if cmd != '' and len(tempo[cmd]) < nb:
                tempo[cmd].append(l[0])
    result = {}
    for cmd in tempo:
        for i, t in enumerate(tempo[cmd]):
            result[cmd + '_t' + str(i+1)] = t
    return result


# In[ ]:


def ht_freq(game):
    counter = Counter()
    for l in game[2:]:
        if len(l)>= 2:
            if l[1] == 'ControlGroupEvent':
                elem = (l[2] + l[3]).replace('\n','')
                counter.update(['ht_' + elem])
    total = sum(counter.values())
    return {k:counter[k] / total * 1000 for k in counter}


# In[ ]:


def nb_selected(game):
    counter = Counter()
    for l in game[2:]:
        if len(l)>= 2:
            if l[1] == 'SelectionEvent':
                elems = l[2].replace('\n','').split(';')
                counter.update(elems)
    counter.pop('', None)
    return {k+'_nbsl':counter[k] for k in counter}


# In[ ]:


def first_n_builds(game, n):
    cpt = 0
    result = {}
    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'BasicCommandEvent':
                cmd = l[2]
            elif l[1] == 'TargetPointCommandEvent':
                cmd = l[-1]
            elif l[1] == 'TargetUnitCommandEvent':
                cmd = l[2]
            cmd = cmd.replace('\n','')
            if cmd.startswith('Build'):
                cpt = cpt + 1
                result['b' + str(cpt)] = l[0]
        if cpt == n:
            break
    return result


# In[ ]:


def nb_workers(game):
    cpt = 0
    race = game[1][0]
    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'BasicCommandEvent':
                cmd = l[2].replace('\n','')
                if race == 'Protoss' and cmd == 'TrainProbe'                     or race == 'Zerg' and cmd == 'MorphDrone'                    or race == 'Terran' and cmd == 'TrainSCV':
                    cpt = cpt + 1
    return {'nb_w' : cpt}


# In[ ]:


def first_build(game):
    d = {}
    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'BasicCommandEvent':
                cmd = l[2]
            elif l[1] == 'TargetPointCommandEvent':
                cmd = l[-1]
            elif l[1] == 'TargetUnitCommandEvent':
                cmd = l[2]
            cmd = cmd.replace('\n','')
            if cmd.startswith('Build') and cmd not in d:
                d[cmd+'_bft'] = l[0]
    return d


# In[ ]:


def motor_skills(game):
    i = -1
    last_t = None
    while len(game[i]) < 2:
        i = i - 1
    try:
        last_t = int(game[i][0]) / (60*16)
    except:
        print(game)
    result = {}
    tot_acts  = len(game[2:])
    
    tot_cam_actions = 0
    
    last_coords = np.array([0,0])
    tot_cam_distance = 0
    
    tot_basic = 0
    
    tot_targetunit = 0
    
    tot_targetpoint = 0
    
    tot_htkey = 0
    
    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'CameraEvent':
                tot_cam_actions += 1
                new_coords = np.array([float(e) for e in l[2:]])
                tot_cam_distance += np.sqrt(np.sum((new_coords - last_coords))**2)
                last_coords = new_coords
            elif l[1] == 'BasicCommandEvent':
                tot_basic += 1
            elif l[1] == 'TargetUnitCommandEvent':
                tot_targetunit += 1
            elif l[1] == 'TargetPointCommandEvent':
                tot_targetpoint += 1
            elif l[1] == 'ControlGroupEvent':
                tot_htkey =+ 1
    return {'apm':tot_acts / last_t, 'cam' : tot_cam_actions / last_t, 'camd' : tot_cam_distance,               'hpm':tot_htkey/last_t, 'bpm':tot_basic/last_t, 'upm':tot_targetunit/last_t, 'ppm':tot_targetpoint/ last_t}
    


# In[ ]:


def buildings_stats(game):
    df = pd.read_csv('./bstats.csv')
    cmds = list(df['building'])
    i = -1
    last_t = None
    while len(game[i]) < 2:
        i = i - 1
    try:
        last_t = int(game[i][0]) / (60*16)
    except:
        exit(0)
    tot_minerals = 0
    tot_gas = 0
    tot_time = 0
    tot_hp = 0
    tot_plasma_sheild = 0
    tot_buildings = 0

    for l in game[2:]:
        if len(l) >= 2:
            cmd = ''
            if l[1] == 'BasicCommandEvent':
                cmd = l[2]
            elif l[1] == 'TargetPointCommandEvent':
                cmd = l[-1]
            elif l[1] == 'TargetUnitCommandEvent':
                cmd = l[2]
        cmd = cmd.replace('\n','')
        if cmd in cmds:
            rw = df[df['building'] == cmd].iloc[0]
            tot_buildings += 1
            rw = df[df['building'] == cmd].iloc[0]
            tot_minerals += rw['mineral']
            tot_gas += rw['gas']
            tot_time += rw['game_speed']
            tot_hp += rw['hp']
            tot_plasma_sheild += rw['PS']
    return {'tot_minerals_b':tot_minerals, 'tot_gas_b':tot_gas, 'tot_time_b':tot_time, 'tot_hp_b':tot_hp,    'tot_plasma_sheild_b':tot_plasma_sheild, 'tot_buildings':tot_buildings, 'buildpm':tot_buildings/last_t}


# In[ ]:


def ht_first(game):
    result = {}
    for l in game[2:]:
        if len(l)>= 2:
            if l[1] == 'ControlGroupEvent':
                elem = (l[2] + l[3]).replace('\n','')
                if 'ht_' + elem + '_ft' not in result:
                    result['ht_' + elem + '_ft'] = l[0]
    return result


# In[ ]:


def game_duration(game):
    if len(game) > 3:
        last_t = None
        i = -1
        while len(game[i]) < 2:
            i = i - 1
        try:
            last_t = int(game[i][0]) / (16)
        except:
            print(game)
            result = {}
        return {'duration' : last_t}
    else:
        return {'duration' : 0}


#       

#      

# **GAMES DURATION**

# In[ ]:


df_duration = pd.read_csv('/kaggle/input/features/duration.csv')


# In[ ]:


df_duration['duration'].describe()


# This summary suggets that 75% of games last more than 14 minutes and 50% of games range between 14 minutes and 21 minutes. 
# 
# Very long/short games may be problematic because short ones (That are due to a connexion problem or a player quiting the game for example) wont contain or capture the main specific player behavior and very long ones will obsecure it since very long games are a rare situation in which players will need to imporovise and take non familiar actions/decisions 

# In[ ]:


nb_games = df_duration.shape[0]
print(sum(df_duration['duration'] < 600) / nb_games)
print(sum(df_duration['duration'] > 2000) / nb_games)


# We can observe that 85% of games are ranging between 10 and 30 minutes which is seemingly plausible but more tests should be done to assess *minumum* and *maximum* duration that are relevant for players identification

# In[ ]:


ax = sn.distplot(df_duration['duration'], rug=True)
_ = ax.set(ylabel='frequency', xlabel='duration (seconds)')


# Game duration appears to follow a normal distribution centred around its mean 1000 seconds and a standard deviation around 475 seconds

# **Number of games per player**

# In[ ]:


df_p_r = pd.read_csv('/kaggle/input/features/p_r.csv')


# In[ ]:


df_player_race = pd.read_csv('/kaggle/input/features/p_r_ht.csv')


# In[ ]:


df_p_r['player'].value_counts().describe()


# In[ ]:


ax = sn.distplot(df_p_r['player'].value_counts(), rug=True)
_ = ax.set(ylabel='frequency', xlabel='number of games')


# Data seems very unbalanced as we see in the above graph. Some players have very few (4, 5 or 6) games and for some others we have more than 30 games. This may be problematic since some players would not be well represented and classifications would be biased towards majority players. 
# 
# In our training data, 50% of players are represented by 7 through 20 games. Tests should be done to assess the accurracy of players identification versus the number of games of each players. 

# **Group keys manipulation analysis**

# Based on the resarch paper *https://dl.acm.org/citation.cfm?id=2702429* (Masters of Control: Behavioral Patterns of Simultaneous Unit Group Manipulation in StarCraft 2) we see that hotkeys use is player specific and can be used to identify him with a very high accurracy given a minimum number of games per player.

# In[ ]:


df_htf = pd.read_csv('/kaggle/input/features/h_0_10_freq.csv')


# In[ ]:


def get_player_freqs(player):
    return df_p_r[df_p_r['player'] ==player].sample(20).join(df_htf).drop(columns=['player','race']).reset_index(drop=True)


# In[ ]:


players = ['http://kr.battle.net/sc2/en/profile/2343733/1/sOs/','http://us.battle.net/sc2/en/profile/3202227/1/viOLet/','http://kr.battle.net/sc2/en/profile/2344333/1/Leenock/','http://kr.battle.net/sc2/en/profile/2343607/1/Rain/','http://eu.battle.net/sc2/en/profile/3538115/1/Golden/','http://kr.battle.net/sc2/en/profile/2344987/1/Life/','http://kr.battle.net/sc2/en/profile/2343479/1/True/','http://kr.battle.net/sc2/en/profile/2343531/1/DREAM/','http://kr.battle.net/sc2/en/profile/2343012/1/Maru/']
f1 = get_player_freqs(players[0])
f2 = get_player_freqs(players[1])
f3 = get_player_freqs(players[2])
f4 = get_player_freqs(players[3])
f5 = get_player_freqs(players[4])
f6 = get_player_freqs(players[5])
f7 = get_player_freqs(players[6])
f8 = get_player_freqs(players[7])
f9 = get_player_freqs(players[8])


# In[ ]:


fig, axs = plt.subplots(3, 3, figsize=(15,15))
fig.tight_layout()
_ = sn.heatmap(f1, ax=axs[0, 0]).set_title(players[0])
_ = sn.heatmap(f2, ax=axs[0, 1]).set_title(players[1])
_ = sn.heatmap(f3, ax=axs[0, 2]).set_title(players[2])
_ = sn.heatmap(f4, ax=axs[1, 0]).set_title(players[3])
_ = sn.heatmap(f5, ax=axs[1, 1]).set_title(players[4])
_ = sn.heatmap(f6, ax=axs[1, 2]).set_title(players[5])
_ = sn.heatmap(f7, ax=axs[2, 0]).set_title(players[6])
_ = sn.heatmap(f8, ax=axs[2, 1]).set_title(players[7])
_ = sn.heatmap(f9, ax=axs[2, 2]).set_title(players[8])


# These heatmap graphs are drawn to check if a player has similar hotkeys use behavior accross most of his games. We plot them for 9 different players and 20 games for each player as indicated above each graph.
# 
# We can observe that each player has almost the same use of each key in each of his games with some differences.
# 
# In contrast, we can observe as well that these keyword features are different from a players to another. For example *.../viOLet/* is the only player among the 9 players that does not use h1, *.../Golden/* and *.../DREAM/*
# are the only two tht use h6 significantly...
# 

# To verify this features' relevance (hotkeys use) we visualize a heatmap dendrogram clustering. 

# In[ ]:


new_df = pd.concat([df_p_r['player'], df_htf], axis=1)
new_df = new_df[new_df['player'].apply(func=lambda p: p in players)].sort_values('player').reset_index(drop=True)


# In[ ]:


sn.clustermap(new_df.drop(columns=['player']), method='average')


# This graph shows that within this group of players, features h7, h8 and h9 are not very relvant to separate them. We see constituting as well a bunch of big clusters that represent each player. This suggests that hotkeys use per player is a discriminant factor that can be used in order to identify players.
# 

# **Other features**

# In addition to hotkeys use we tried to extract other features like: motor skills, different buildings stats, number of workers, first time of hotkeys use, first time of building (To capture the builfing tech tree behavior) etc...

#     

# **Prediction**

# In[ ]:


df_htf_all = pd.read_csv('/kaggle/input/features/ht_freq.csv')
df_mtr = pd.read_csv('/kaggle/input/features/mtskills.csv')
df_first_ht = pd.read_csv('/kaggle/input/features/hf.csv')


# In[ ]:


df_mtr


# In[ ]:


df = pd.concat([df_p_r, df_htf_all, df_mtr], axis=1)


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


idx = np.arange(df.shape[0])
np.random.shuffle(idx)
training = idx[:df.shape[0] - 400]
test = idx[df.shape[0] - 400:]


# In[ ]:


df_training = df.iloc[training]
df_test = df.iloc[test]
dpths = [5, 10, 15, 20, 25, 30, 35]
nests = [20, 30, 50, 100, 150, 200, 250, 300]
results2 = np.ndarray(shape=(len(dpths), len(nests)))
for i, d in enumerate(dpths):
    for j, nest in enumerate(nests):
        clf = RandomForestClassifier(n_estimators=nest, max_depth=d)
        clf.fit(df_training.iloc[:, 2:], df_training['player'])
        pred_labels = clf.predict(df_test.iloc[:, 2:])    
        results2[i,j] = f1_score(df_test['player'], pred_labels, average='macro')


# In[ ]:


htmdf = pd.DataFrame(results2, columns=['ne' + str(e) for e in nests])
htmdf.index = ['d' + str(e) for e in dpths]
sn.heatmap(htmdf)


# This heat map suggests that small trees' depths are very constaining and therefor lead to bad performance. Increasing the depth is followed by an increase in the prediction accuracy but at a certain point increasing this parameter will have no effect on performance. Same is to be said about the number of estimators parameter.
# 
# One can choose the configuratio nwith the best results but here we opted for a validation set approach with is not very precise so cross validation would be more usefull.

# **Confusion matrix**

# In[ ]:


clf = RandomForestClassifier(n_estimators=250, max_depth=25)
clf.fit(df_training.iloc[:, 2:], df_training['player'])
pred_labels = clf.predict(df_test.iloc[:, 2:])    


# In[ ]:


f1_score(df_test['player'], pred_labels, average='macro')


# In[ ]:


cm = confusion_matrix(df_test['player'], pred_labels)


# In[ ]:


for i in range(0, cm.shape[0]):
    row = cm[i, :]
    sm = sum(row)
    if sm != 0:
        cm[i, :] = row / sm


# In[ ]:


fig= plt.figure(figsize=(16,8))
sn.heatmap(cm)


# This graph suggests that our model is not very bad even though f1 score is just 80% given that when our model incorrectly classifies a player it predicts just another play. This confirm our hypothesis about players who are playing using different usernames (At least different URLs in the dataset) and so the URL is not a primary key for our records. 

# In[ ]:




