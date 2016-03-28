<?php

$timePoint = 300;
$suffix = '_test';
$parseItems = false;
$parseAbilities = false;
$parseObj = false;
$parseBB = false;

if ($parseItems) {
    $items = getItems();
    $itemsRowTpl = [
        'match_id' => 0
    ];

    foreach (['r', 'd'] as $team) {
        foreach ($items as $id => $name) {
            $itemsRowTpl["{$team}_{$name}_count"] = 0;
        }
    }

    $itemsFile = new SplFileObject(__DIR__ . "/data/items{$suffix}.csv", 'w+');
    $itemsFile->fputcsv(array_keys($itemsRowTpl));
}

if ($parseAbilities) {
    $abilities = getAbilities();
    $abilitiesRowTpl = [
        'match_id' => 0
    ];

    foreach (['r', 'd'] as $team) {
        foreach ($abilities as $id => $name) {
            $abilitiesRowTpl["{$team}_{$name}_level"] = 0;
        }
    }

    $abilitiesFile = new SplFileObject(__DIR__ . "/data/abilities{$suffix}.csv", 'w+');
    $abilitiesFile->fputcsv(array_keys($abilitiesRowTpl));
}

if ($parseObj) {
    $objRowTpl = [
        'match_id' => 0,
    ];

    foreach (['r', 'd'] as $team) {
        foreach (['tower_kill', 'tower_deny', 'roshan_kill'] as $name) {
            $objRowTpl["{$team}_{$name}_count"] = 0;
        }
    }

    $obfFile = new SplFileObject(__DIR__ . "/data/obj{$suffix}.csv", 'w+');
    $obfFile->fputcsv(array_keys($objRowTpl));
}

if ($parseBB) {
    $bbRowTpl = [
        'match_id' => 0,
        'r_bb_count' => 0,
        'd_bb_count' => 0,
    ];

    $bbFile = new SplFileObject(__DIR__ . "/data/bb{$suffix}.csv", 'w+');
    $bbFile->fputcsv(array_keys($bbRowTpl));
}

$file = new SplFileObject(__DIR__ . "/data/matches{$suffix}.jsonlines");
while ($line = $file->fgets()) {
    if ($data = json_decode($line, true)) {
        echo "{$data['match_id']}\n";

        if ($parseItems) {
            $itemsRow = $itemsRowTpl;
            $itemsRow['match_id'] = $data['match_id'];
        }

        if ($parseAbilities) {
            $abilitiesRow = $abilitiesRowTpl;
            $abilitiesRow['match_id'] = $data['match_id'];
        }

        if ($parseBB) {
            $bbRow = $bbRowTpl;
            $bbRow['match_id'] = $data['match_id'];
        }

        if ($parseObj) {
            $objRow = $objRowTpl;
            $objRow['match_id'] = $data['match_id'];

            foreach ($data['objectives'] as $obj) {
                if ($obj['time'] <= $timePoint) {
                    if (isset($obj['player1'])) {
                        $n = $obj['player1'];
                        if ($n < 5) {
                            $teamIx = 'r';
                            $playerIx = $n + 1;
                        } else {
                            $teamIx = 'd';
                            $playerIx = $n - 4;
                        }

                        $key = "{$teamIx}_{$obj['type']}_count";
                        if (isset($objRow[$key])) {
                            $objRow[$key] += 1;
                        }
                    }
                } else {
                    break;
                }
            }
        }


        foreach ($data['players'] as $n => $player) {
            if ($n < 5) {
                $teamIx = 'r';
                $playerIx = $n + 1;
            } else {
                $teamIx = 'd';
                $playerIx = $n - 4;
            }

            if ($parseBB) {
                foreach ($player['buyback_log'] as $bb) {
                    if ($bb['time'] <= $timePoint) {
                        $bbRow[$teamIx . '_bb_count'] += 1;
                    }
                }
            }

            if ($parseItems) {
                foreach ($player['purchase_log'] as $purchase) {
                    if ($purchase['time'] <= $timePoint) {
                        $itemName = $items[$purchase['item_id']];
                        $itemsRow["{$teamIx}_{$itemName}_count"] += 1;
                    } else {
                        break;
                    }
                }
            }

            if ($parseAbilities) {
                foreach ($player['ability_upgrades'] as $upgrade) {
                    if ($upgrade['time'] <= $timePoint) {
                        if (isset($abilities[$upgrade['ability']])) {
                            $abilityName = $abilities[$upgrade['ability']];
                            $abilitiesRow["{$teamIx}_{$abilityName}_level"] += 1;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        if ($parseItems) {
            $itemsFile->fputcsv($itemsRow);
        }

        if ($parseAbilities) {
            $abilitiesFile->fputcsv($abilitiesRow);
        }

        if ($parseObj) {
            $obfFile->fputcsv($objRow);
        }

        if ($parseBB) {
            $bbFile->fputcsv($bbRow);
        }
    }
}

function getItems()
{
    $file = new SplFileObject(__DIR__ . "/data/dictionaries/items.csv");
    $items = [];

    while ($line = $file->fgetcsv()) {
        if (is_numeric($line[0])) {
            $items[$line[0]] = $line[1];
        }
    }

    return $items;
}

function getAbilities()
{
    $file = new SplFileObject(__DIR__ . "/data/dictionaries/abilities.csv");
    $abilities = [];

    while ($line = $file->fgetcsv()) {
        if (is_numeric($line[0])) {
            $abilities[$line[0]] = $line[1];
        }
    }

    return $abilities;
}
