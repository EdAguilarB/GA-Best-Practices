from rdkit import Chem

iso_smarts = Chem.MolFromSmarts("[N]#[C;D1]")
primary_amine_smarts = Chem.MolFromSmarts("[NX3;H2][#6]")
# Ugi's carbonyl component is an aldehyde (1 H on the carbonyl C) or a ketone
# (0 H, two carbon substituents). Requiring two carbons on the ketone branch keeps
# acids, esters and amides out.
carbonyl_smarts = Chem.MolFromSmarts("[CX3;H1,$([CX3H0]([#6])[#6])]=O")
carboxy_smarts = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")


def smiles_to_mol(smiles):
    """Parse a SMILES string into an editable RWMol."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.RWMol(mol)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _unique_match(mol, smarts, name):
    """Return the single substructure match for `smarts`, or None if there
    isn't exactly one."""
    matches = mol.GetSubstructMatches(smarts)
    if len(matches) != 1:
        print(f"{len(matches)} matches were found for the {name}. Only one was expected")
        return None
    return matches[0]


def _live_attachments(mol):
    """Indices of atoms currently flagged as active attachment points."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.HasProp("Attach") and a.GetBoolProp("Attach")]


def _join_on_attachments(base, fragment, deactivate_base=True):
    """Combine `base` and `fragment`, bond their two live attachment points,
    and (optionally) retire the one coming from `base` so the next coupling
    targets the freshly added fragment.

    Because `base` is combined first, its attachment always has the lower
    index, i.e. it is `live[0]` -- so `deactivate_base` deterministically
    retires the base atom (this is what the old TODO was worried about).
    """
    rw = Chem.RWMol(Chem.CombineMols(base, fragment))
    live = _live_attachments(rw)
    assert len(live) == 2
    rw.AddBond(live[0], live[1], Chem.BondType.SINGLE)
    if deactivate_base:
        rw.GetAtomWithIdx(live[0]).SetBoolProp("Attach", False)
    return rw


# ---------------------------------------------------------------------------
# Per-reactant preparation: tag the attachment atom(s) and strip leaving groups
# ---------------------------------------------------------------------------
def _prepare_acid(acid):
    """Drop the -OH of the carboxylic acid; flag the carbonyl C for bonding."""
    match = _unique_match(acid, carboxy_smarts, "carboxylic acid")
    if match is None:
        return None
    carbon = None
    for idx in match:
        atom = acid.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            atom.SetBoolProp("Attach", True)
            carbon = idx
        # The OH oxygen is the one with a hydrogen neighbour (total degree 2)
        elif atom.GetSymbol() == "O" and atom.GetTotalDegree() == 2:
            acid.RemoveBond(idx, carbon)  # remove bond to carbon first
            acid.RemoveAtom(idx)  # then remove the OH oxygen
            break
    return acid


def _prepare_amine(amine, keep_hydrogens=0):
    """Flag the primary-amine N for bonding and fix its hydrogen count.

    The four-component reaction acylates this nitrogen and also bonds it to the
    carbonyl carbon, so it keeps no hydrogens. Without an acid there is only one
    new bond to make, and the nitrogen ends up a secondary amine, so one
    hydrogen has to stay.
    """
    match = _unique_match(amine, primary_amine_smarts, "primary amine")
    if match is None:
        return None
    for idx in match:
        atom = amine.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N":
            atom.SetBoolProp("Attach", True)
            atom.SetNumExplicitHs(keep_hydrogens)
            atom.SetNoImplicit(True)
            break
    return amine


def _prepare_carbonyl(aldehyde):
    """Drop the carbonyl O; flag the (former carbonyl) C for bonding.
    Works for both aldehydes and ketones."""
    match = _unique_match(aldehyde, carbonyl_smarts, "carbonyl (aldehyde or ketone)")
    if match is None:
        return None
    C_ald = O_ald = None
    for idx in match:
        atom = aldehyde.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            atom.SetBoolProp("Attach", True)
            C_ald = idx
        elif atom.GetSymbol() == "O":
            O_ald = idx
    aldehyde.RemoveBond(C_ald, O_ald)
    aldehyde.RemoveAtom(O_ald)
    return aldehyde


def _prepare_isocyanide(isocyanide):
    """Turn R-N#C into R-NH-C(=O)-; flag the (new carbonyl) C for bonding."""
    match = _unique_match(isocyanide, iso_smarts, "isocyanide")
    if match is None:
        return None
    C_iso = N_iso = None
    for idx in match:
        atom = isocyanide.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            atom.SetFormalCharge(0)
            atom.SetBoolProp("Attach", True)
            C_iso = idx
        elif atom.GetSymbol() == "N":
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(1)
            N_iso = idx
    # triple bond -> single bond, then add the carbonyl oxygen
    isocyanide.RemoveBond(C_iso, N_iso)
    isocyanide.AddBond(C_iso, N_iso, Chem.BondType.SINGLE)
    o_idx = isocyanide.AddAtom(Chem.Atom("O"))
    isocyanide.AddBond(C_iso, o_idx, Chem.BondType.DOUBLE)
    return isocyanide


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def get_ugi_product(primary_amine, aldehyde, isocyanide, carboxylic_acid=None):
    """Assemble a Ugi product. Omitting carboxylic_acid runs the
    three-component variant, which leaves the amine nitrogen secondary rather
    than acylating it to an amide."""
    three_component = carboxylic_acid is None

    # Without an acid the nitrogen makes only one new bond, so it must retain a
    # hydrogen instead of being stripped bare for two.
    amine = _prepare_amine(smiles_to_mol(primary_amine), keep_hydrogens=int(three_component))
    ald = _prepare_carbonyl(smiles_to_mol(aldehyde))
    iso = _prepare_isocyanide(smiles_to_mol(isocyanide))
    acid = None if three_component else _prepare_acid(smiles_to_mol(carboxylic_acid))

    required = (amine, ald, iso) if three_component else (acid, amine, ald, iso)
    if None in required:
        return

    # Chain the couplings. The amine N is the hub: with an acid it bonds that
    # carbonyl first, then the aldehyde carbon; the aldehyde carbon finally
    # bonds the isocyanide carbon.
    if three_component:
        rw = _join_on_attachments(amine, ald)  # amine N --  aldehyde C
    else:
        rw = _join_on_attachments(acid, amine)  # acid C  --  amine N
        rw = _join_on_attachments(rw, ald)  # amine N --  aldehyde C
    rw = _join_on_attachments(rw, iso, deactivate_base=False)  # aldehyde C -- isocyanide C

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol
