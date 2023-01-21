import { Box } from "@mantine/core";

interface Props {
  white: number;
}

export const Indicator = ({ white }: Props) => {
  return (
    <Box w="30px" px="sm">
      <Box h={`${(1 - white) * 100}%`} bg="dark"></Box>
      <Box h={`${white * 100}%`} bg="gray"></Box>
    </Box>
  );
};
