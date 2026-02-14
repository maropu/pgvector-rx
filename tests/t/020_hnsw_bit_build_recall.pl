use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node;
my @queries = ();
my @expected;
my $limit = 20;
my $dim = 52;
my $max = 2**$dim;

sub test_recall
{
	my ($min, $operator) = @_;
	my $correct = 0;
	my $total = 0;

	my $explain = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SET hnsw.ef_search = 100;
		EXPLAIN ANALYZE SELECT i FROM tst ORDER BY v $operator $queries[0] LIMIT $limit;
	));
	like($explain, qr/Index Scan/);

	for my $i (0 .. $#queries)
	{
		my $actual = $node->safe_psql("postgres", qq(
			SET enable_seqscan = off;
			SET hnsw.ef_search = 100;
			SELECT i FROM tst ORDER BY v $operator $queries[$i] LIMIT $limit;
		));
		my @actual_ids = split("\n", $actual);

		my @expected_ids = split("\n", $expected[$i]);
		my %expected_set = map { $_ => 1 } @expected_ids;

		foreach (@actual_ids)
		{
			if (exists($expected_set{$_}))
			{
				$correct++;
			}
		}

		$total += $limit;
	}

	cmp_ok($correct / $total, ">=", $min, $operator);
}

# Initialize node
$node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

# Create table
$node->safe_psql("postgres", "CREATE EXTENSION pgvector_rx;");
$node->safe_psql("postgres", "CREATE TABLE tst (i int4, v bit($dim));");
$node->safe_psql("postgres",
	"INSERT INTO tst SELECT i, (random() * $max)::bigint::bit($dim) FROM generate_series(1, 10000) i;"
);

# Generate queries
for (1 .. 20)
{
	my $r = int(rand() * $max);
	push(@queries, "${r}::bigint::bit($dim)");
}

# Check each index type (sequential build only)
my @operators = ("<~>", "<\%>");
my @opclasses = ("bit_hamming_ops", "bit_jaccard_ops");

for my $i (0 .. $#operators)
{
	my $operator = $operators[$i];
	my $opclass = $opclasses[$i];

	# Get exact results
	@expected = ();
	foreach (@queries)
	{
		# Handle ties
		my $res = $node->safe_psql("postgres", qq(
			WITH top AS (
				SELECT v $operator $_ AS distance FROM tst ORDER BY distance LIMIT $limit
			)
			SELECT i FROM tst WHERE (v $operator $_) <= (SELECT MAX(distance) FROM top)
		));
		push(@expected, $res);
	}

	# Build index serially
	$node->safe_psql("postgres", qq(
		SET max_parallel_maintenance_workers = 0;
		CREATE INDEX idx ON tst USING hnsw (v $opclass);
	));

	# Test approximate results
	my $min = $operator eq "<\%>" ? 0.95 : 0.98;
	test_recall($min, $operator);

	$node->safe_psql("postgres", "DROP INDEX idx;");
}

done_testing();
